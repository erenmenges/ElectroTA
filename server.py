import os
import asyncio
import struct
import math
import json
import base64
import time
import traceback
import logging
from collections import deque

from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

logging.basicConfig(level=logging.WARNING)
load_dotenv()

api_key1 = os.getenv("API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
client = genai.Client(api_key=api_key1)
model = "gemini-2.5-flash-native-audio-preview-12-2025"

# ─────────────────────────────────────────────
# Define the probe function for Gemini
# ─────────────────────────────────────────────
probe_declaration = {
    "name": "probe",
    "description": (
        "Reads voltage and current measurements from the probe instrument. "
        "Returns voltage (V) and current (A) readings. "
        "\n**Invocation Condition:** ONLY UNMISTAKABLY call this function when the user "
        "literally says the word 'probe'. Do NOT call it for theoretical "
        "questions, calculations, or general discussion about voltages or "
        "currents. The user must explicitly say 'probe' to trigger a reading."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

tools = [{"function_declarations": [probe_declaration]}]

DETAILED_PROMPT = (
    "Analyze this circuit schematic completely. Return ONLY the "
    "following sections, no other text.\n\n"
    "COMPONENTS:\n"
    "List every component with its type, label, value (if shown), "
    "and the nodes it connects. For 2-terminal devices use "
    "NODE_A - NODE_B. For 3+ terminal devices (transistors, op-amps), "
    "list all terminals.\n"
    "Format: LABEL | TYPE | VALUE | NODES\n\n"
    "NODES:\n"
    "List every unique node and what connects to it.\n\n"
    "POWER RAILS:\n"
    "List any labeled supply voltages (VCC, VDD, +5V, +12V, etc.) "
    "with their voltage value and the node they connect to.\n"
    "Format: LABEL | VALUE | NODE\n"
    "If none, write \"None.\"\n\n"
    "LABELED VOLTAGES:\n"
    "List every voltage explicitly marked on the schematic with "
    "its polarity (+ and - terminals).\n"
    "Format: LABEL | POSITIVE_NODE - NEGATIVE_NODE\n\n"
    "IMPLIED VOLTAGES:\n"
    "List voltages that exist across components but are NOT labeled "
    "on the schematic. Every component has a voltage across it.\n"
    "Format: LABEL | POSITIVE_NODE - NEGATIVE_NODE\n\n"
    "CURRENT PATHS:\n"
    "Describe each current loop. For each loop list the direction "
    "(CW/CCW) and the components in order. Note any current "
    "direction arrows shown on the schematic.\n\n"
    "GROUND REFERENCE:\n"
    "Which node is ground or the reference node? If not explicitly "
    "marked, state \"Not marked.\"\n\n"
    "POLARITY NOTES:\n"
    "For any component where orientation matters (diodes, LEDs, "
    "electrolytic caps, transistors, op-amps), state the orientation "
    "explicitly (e.g., \"D1 anode at NODE_X, cathode at NODE_Y\", "
    "\"Q1 collector at NODE_X, base at NODE_Y, emitter at NODE_Z\", "
    "\"U1 non-inverting input at NODE_X, inverting input at NODE_Y, "
    "output at NODE_Z\").\n"
    "If none, write \"None.\"\n\n"
    "KEY RELATIONSHIPS:\n"
    "List any KVL, KCL, or voltage divider equations that are "
    "immediately obvious from the topology."
)

COMPRESS_PROMPT = (
    "Compress this circuit analysis into the most compact notation "
    "possible while preserving all information. Use plain ASCII only, "
    "but make it readable."
)

REMINDER_INTERVAL = 60
IMAGE_COOLDOWN_SECONDS = 5
SESSION_MAX_IMAGES = 20
SESSION_MAX_DURATION = 20 * 60  # 20 minutes hard limit


async def analyze_circuit_image(image_bytes, mime_type):
    """Two-call pipeline: detailed analysis, then compression."""
    response1 = await client.aio.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            DETAILED_PROMPT,
        ],
    )
    detailed = response1.text

    response2 = await client.aio.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[detailed, COMPRESS_PROMPT],
    )
    compact = response2.text

    return detailed, compact


def build_config(detailed_analysis):
    """Build the live session config with the detailed analysis in the system prompt."""
    system_instruction = (
        "**Persona:** You are ElectroTA, a helpful electronics lab teaching assistant. "
        "You are helping a student who is learning electronics and building circuits. "
        "Students make mistakes — that is expected and that is why you exist.\n\n"
        "**Target Circuit (the goal):**\n"
        "The circuit below is what the student is TRYING to build. It is the correct, "
        "intended design. The student's actual wiring may or may not match this.\n"
        "```\n"
        f"{detailed_analysis}\n"
        "```\n\n"
        "**Conversational Rules:**\n"
        "1. Help the student understand electronics concepts and guide them through "
        "building and debugging the target circuit above.\n"
        "2. ONLY call the `probe` function when the student literally says the word "
        "'probe'. Do NOT call it for theoretical questions, calculations, or general "
        "discussion about voltages or currents. If the student asks 'what is the voltage "
        "at X?' without saying 'probe', answer theoretically using circuit analysis.\n"
        "3. After receiving probe results, read the voltage and current values "
        "back to the student clearly.\n\n"
        "**Image Analysis Rules (CRITICAL):**\n"
        "When the student sends an image, follow these steps strictly:\n"
        "Step 1 — IDENTIFY: What is the image? Is it a photo of a breadboard/circuit, "
        "an oscilloscope screen, a schematic, or something else entirely? If it is NOT "
        "electronics-related (e.g., people, a room, text), say what you see and ask "
        "the student how it relates.\n"
        "Step 2 — DESCRIBE INDEPENDENTLY: If it IS a circuit photo, describe the "
        "physical wiring you actually see — which components are where, what connects "
        "to what, which breadboard rows are used. Do this WITHOUT referencing the target "
        "circuit above. Pretend you have never seen the target schematic.\n"
        "Step 3 — COMPARE AND FIND ERRORS: Only after completing Step 2, compare your "
        "independent description against the target circuit node by node. Call out every "
        "discrepancy. Assume errors are likely — the student is learning and may have "
        "wired something wrong. Be specific: name which wire is in the wrong place and "
        "where it should go.\n"
        "NEVER skip Steps 1-2 and jump straight to 'looks correct'. NEVER assume the "
        "student's wiring matches the target.\n\n"
        "**Guardrails:**\n"
        "- Never invent or estimate measurement data.\n"
        "- Never fabricate what an image contains — describe only what you actually see.\n"
        "- When in doubt about wiring, say so. It is far better to flag a possible "
        "error than to miss a real one."
    )
    return {
        "response_modalities": ["AUDIO"],
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": True},
        },
        "output_audio_transcription": {},
        "input_audio_transcription": {},
        "context_window_compression": {
            "sliding_window": {},
        },
        "tools": tools,
        "system_instruction": system_instruction,
    }

INPUT_AUDIO_RATE = 16000
CHUNK_SIZE = 1024

# --- VAD tuning knobs ---
INTERRUPT_MULTIPLIER = 3.0
MIN_INTERRUPT_RMS = 800
QUIET_SPEECH_RMS = 200
SILENCE_CHUNKS_TO_END = 14
INTERRUPT_CONFIRM_CHUNKS = 2
PRE_BUFFER_CHUNKS = 16
MAX_UTTERANCE_SECONDS = 50

app = FastAPI()
_session_lock = asyncio.Lock()


class ConnectionRateLimiter:
    """Limits how many times a given IP can trigger the analysis pipeline."""

    def __init__(self, max_calls, window_seconds):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps = {}  # ip -> list of monotonic timestamps

    def is_allowed(self, ip):
        now = time.monotonic()
        if ip not in self._timestamps:
            self._timestamps[ip] = []

        # Drop timestamps outside the window
        self._timestamps[ip] = [
            t for t in self._timestamps[ip] if now - t < self.window_seconds
        ]

        if len(self._timestamps[ip]) >= self.max_calls:
            return False

        self._timestamps[ip].append(now)
        return True


# Allow at most 3 analysis-triggering connections per IP per 60 seconds
_analysis_rate_limiter = ConnectionRateLimiter(max_calls=3, window_seconds=60)


def compute_rms(audio_bytes):
    if len(audio_bytes) < 2:
        return 0.0
    n_samples = len(audio_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", audio_bytes[: n_samples * 2])
    if n_samples == 0:
        return 0.0
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / n_samples)


class SessionState:
    def __init__(self):
        self.is_playing = False
        self.speaker_rms = 0.0
        self.user_is_active = False
        self.activity_start_time = 0.0
        self.interrupt_requested = False
        self.silence_counter = 0
        self.interrupt_confirm_counter = 0
        self.interrupt_audio_buffer = []
        self.pre_buffer = deque(maxlen=PRE_BUFFER_CHUNKS)
        self.probe_event = asyncio.Event()
        self.probe_result = None
        self.disconnected = asyncio.Event()
        self.last_image_time = 0.0
        self.image_count = 0

    def is_loud_enough(self, mic_rms):
        if self.is_playing:
            threshold = max(self.speaker_rms * INTERRUPT_MULTIPLIER, MIN_INTERRUPT_RMS)
        else:
            threshold = QUIET_SPEECH_RMS
        return mic_rms >= threshold


async def process_audio(session, state, ws, audio_data):
    """Apply VAD logic to incoming audio and forward to Gemini as needed."""
    mic_rms = compute_rms(audio_data)
    loud = state.is_loud_enough(mic_rms)

    # Force-end utterance if the user has been "speaking" for too long.
    # This prevents ambient noise from keeping the stream open indefinitely.
    if state.user_is_active:
        elapsed = time.monotonic() - state.activity_start_time
        if elapsed >= MAX_UTTERANCE_SECONDS:
            state.user_is_active = False
            state.silence_counter = 0
            state.activity_start_time = 0.0
            await session.send_realtime_input(
                activity_end=types.ActivityEnd()
            )
            print(
                f"\n[<<< Max utterance cutoff ({MAX_UTTERANCE_SECONDS}s): "
                f"forced activity_end]"
            )
            return

    if state.is_playing and not state.user_is_active:
        state.pre_buffer.append(audio_data)

    if loud:
        state.silence_counter = 0
        if not state.user_is_active:
            if state.is_playing:
                state.interrupt_confirm_counter += 1
                state.interrupt_audio_buffer.append(audio_data)
                if state.interrupt_confirm_counter >= INTERRUPT_CONFIRM_CHUNKS:
                    state.user_is_active = True
                    state.activity_start_time = time.monotonic()
                    state.interrupt_requested = True
                    state.is_playing = False
                    await ws.send_json({"type": "interrupt"})
                    await session.send_realtime_input(
                        activity_start=types.ActivityStart()
                    )
                    for buffered in state.pre_buffer:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=buffered, mime_type="audio/pcm;rate=16000"
                            )
                        )
                    state.pre_buffer.clear()
                    state.interrupt_audio_buffer.clear()
                    state.interrupt_confirm_counter = 0
                    print(
                        f"\n[>>> INTERRUPT: activity_start sent, mic_rms={mic_rms:.0f}]"
                    )
            else:
                state.user_is_active = True
                state.activity_start_time = time.monotonic()
                state.interrupt_confirm_counter = 0
                state.interrupt_audio_buffer.clear()
                state.pre_buffer.clear()
                await session.send_realtime_input(
                    activity_start=types.ActivityStart()
                )
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_data, mime_type="audio/pcm;rate=16000"
                    )
                )
                print(
                    f"\n[>>> Speaking: activity_start sent, mic_rms={mic_rms:.0f}]"
                )
        else:
            await session.send_realtime_input(
                audio=types.Blob(
                    data=audio_data, mime_type="audio/pcm;rate=16000"
                )
            )
    else:
        state.interrupt_confirm_counter = 0
        state.interrupt_audio_buffer.clear()
        if state.user_is_active:
            state.silence_counter += 1
            await session.send_realtime_input(
                audio=types.Blob(
                    data=audio_data, mime_type="audio/pcm;rate=16000"
                )
            )
            if state.silence_counter >= SILENCE_CHUNKS_TO_END:
                state.user_is_active = False
                state.silence_counter = 0
                state.activity_start_time = 0.0
                await session.send_realtime_input(
                    activity_end=types.ActivityEnd()
                )
                print(f"\n[<<< Done speaking: activity_end sent]")


async def process_image(session, state, ws, image_data, mime_type):
    """Send an image to Gemini as an interrupt with a silent audio nudge."""
    print(f"\n[>>> Photo: sending {len(image_data)} bytes as interrupt]")

    if state.is_playing:
        state.interrupt_requested = True
        state.is_playing = False
        await ws.send_json({"type": "interrupt"})

    SILENCE_DURATION_MS = 500
    num_silence_bytes = int(INPUT_AUDIO_RATE * 2 * SILENCE_DURATION_MS / 1000)
    silent_audio = b"\x00" * num_silence_bytes

    await session.send_realtime_input(
        activity_start=types.ActivityStart()
    )
    await session.send_realtime_input(
        video=types.Blob(data=image_data, mime_type=mime_type)
    )
    await session.send_realtime_input(
        audio=types.Blob(data=silent_audio, mime_type="audio/pcm;rate=16000")
    )
    await session.send_realtime_input(
        activity_end=types.ActivityEnd()
    )

    state.user_is_active = False
    print("[>>> Photo sent — Gemini will respond]")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not AUTH_TOKEN:
        await ws.close(code=1008, reason="Server AUTH_TOKEN not configured")
        return

    auth_header = ws.headers.get("authorization", "")
    if auth_header != f"Bearer {AUTH_TOKEN}":
        await ws.close(code=1008, reason="Unauthorized")
        return

    client_ip = ws.client.host if ws.client else "unknown"

    if not _analysis_rate_limiter.is_allowed(client_ip):
        await ws.accept()
        await ws.send_json({
            "type": "error",
            "message": "Rate limit exceeded. Please wait before reconnecting.",
        })
        await ws.close(code=1008, reason="Rate limit exceeded")
        print(f"[Rate limit] Rejected connection from {client_ip}")
        return

    if _session_lock.locked():
        await ws.accept()
        await ws.send_json({
            "type": "error",
            "message": "Another session is already active. Only one client at a time.",
        })
        await ws.close(code=1008, reason="Session limit reached")
        return

    async with _session_lock:
        await ws.accept()
        state = SessionState()
        await _run_session(ws, state)


async def _run_session(ws: WebSocket, state: SessionState):

    print("Client connected. Waiting for circuit diagram...")

    try:
        # Step 1: Wait for the circuit diagram from the client
        raw = await ws.receive_json()
        if raw.get("type") != "circuit_diagram":
            await ws.send_json({"type": "error", "message": "Expected circuit_diagram as first message"})
            await ws.close()
            return

        image_data = base64.b64decode(raw["data"])
        mime_type = raw.get("mime_type", "image/jpeg")
        print(f"Received circuit diagram ({len(image_data)} bytes). Analyzing...")

        # Step 2: Two-call pipeline — detailed analysis then compression
        detailed, compact = await analyze_circuit_image(image_data, mime_type)
        print(f"Detailed analysis:\n{detailed}")
        print(f"\nCompact analysis:\n{compact}")

        # Step 3: Build config with detailed analysis in the system prompt
        session_config = build_config(detailed)

        await ws.send_json({"type": "status", "message": "Circuit analyzed. Starting live session..."})

        # Step 4: Start the live session with the circuit-aware config
        async with client.aio.live.connect(model=model, config=session_config) as session:
            print("Connected to Gemini!")

            # Inject the detailed analysis as an initial briefing turn
            briefing = [
                types.Content(
                    role="user",
                    parts=[types.Part(text="Describe the circuit under test in detail.")],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part(text=detailed)],
                ),
            ]
            await session.send_client_content(turns=briefing, turn_complete=False)
            print("[Briefing injected into session context]")

            await ws.send_json({"type": "status", "message": "Connected to Gemini"})

            # Build the reminder text from the compact analysis
            reminder_text = (
                "[SYSTEM REMINDER — do not respond to this message, do not acknowledge it, "
                "do not mention it. Continue responding to the user normally.]\n"
                "Target circuit (what the student is trying to build):\n"
                f"{compact}\n"
                "Remember: The student's actual wiring may have errors — your job is to catch them.\n"
                "ONLY call probe() when the student literally says the word 'probe'. "
                "Do NOT call it for theoretical questions about voltages or currents.\n"
                "When the student sends an image: (1) identify what it is, (2) if it is a "
                "circuit, describe the physical wiring independently WITHOUT referencing "
                "the target, (3) only then compare against the target node by node and "
                "flag every discrepancy. NEVER skip to 'looks correct'. NEVER assume "
                "the wiring matches the target. Errors are expected."
            )

            async def handle_client_messages():
                """Receive messages from client WebSocket and process them."""
                try:
                    while True:
                        raw = await ws.receive_json()
                        msg_type = raw.get("type")

                        if msg_type == "audio":
                            audio_data = base64.b64decode(raw["data"])
                            await process_audio(session, state, ws, audio_data)

                        elif msg_type == "image":
                            if state.image_count >= SESSION_MAX_IMAGES:
                                await ws.send_json({
                                    "type": "error",
                                    "message": f"Image limit reached ({SESSION_MAX_IMAGES} per session).",
                                })
                                continue
                            now = time.monotonic()
                            if now - state.last_image_time < IMAGE_COOLDOWN_SECONDS:
                                await ws.send_json({
                                    "type": "error",
                                    "message": "Image rate limited. Please wait a few seconds.",
                                })
                                continue
                            state.last_image_time = now
                            state.image_count += 1
                            image_data = base64.b64decode(raw["data"])
                            mime_type = raw.get("mime_type", "image/jpeg")
                            await process_image(session, state, ws, image_data, mime_type)

                        elif msg_type == "probe_result":
                            state.probe_result = raw["data"]
                            state.probe_event.set()
                finally:
                    state.disconnected.set()

            async def handle_gemini_responses():
                """Receive responses from Gemini and forward to client."""
                while True:
                    try:
                        async for response in session.receive():

                            # Handle tool calls from Gemini
                            if response.tool_call:
                                print("\n[Tool call received]")
                                function_responses = []
                                for fc in response.tool_call.function_calls:
                                    print(f"  → Gemini wants to call: {fc.name}()")

                                    if fc.name == "probe":
                                        state.probe_event.clear()
                                        await ws.send_json({"type": "probe_request"})

                                        probe_task = asyncio.create_task(state.probe_event.wait())
                                        dc_task = asyncio.create_task(state.disconnected.wait())
                                        done, pending = await asyncio.wait(
                                            [probe_task, dc_task],
                                            return_when=asyncio.FIRST_COMPLETED,
                                        )
                                        for t in pending:
                                            t.cancel()

                                        if state.disconnected.is_set():
                                            print("\n[Client disconnected while waiting for probe result]")
                                            return

                                        result = state.probe_result
                                        print(f"  ← probe() returned: {result}")
                                    else:
                                        result = {"error": f"Unknown function: {fc.name}"}

                                    function_responses.append(
                                        types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={"result": result},
                                        )
                                    )

                                await session.send_tool_response(
                                    function_responses=function_responses
                                )
                                print(
                                    "  [Tool response sent → Gemini will now speak the results]"
                                )

                            # Handle server content (audio, transcriptions, etc.)
                            server_content = response.server_content
                            if server_content:
                                if server_content.interrupted:
                                    print("\n[Server: Interrupted]")
                                    state.is_playing = False
                                    state.interrupt_requested = False

                                if server_content.turn_complete:
                                    print("\n[Server: Turn complete]")
                                    state.is_playing = False
                                    state.interrupt_requested = False

                                if server_content.output_transcription:
                                    text = server_content.output_transcription.text
                                    if text:
                                        print(f"{text}", end="", flush=True)
                                        await ws.send_json(
                                            {"type": "transcript_output", "text": text}
                                        )

                                if server_content.input_transcription:
                                    text = server_content.input_transcription.text
                                    if text:
                                        print(
                                            f"\n  [You]: {text}",
                                            end="",
                                            flush=True,
                                        )
                                        await ws.send_json(
                                            {"type": "transcript_input", "text": text}
                                        )

                                if server_content.model_turn:
                                    for part in server_content.model_turn.parts:
                                        if part.inline_data:
                                            if state.interrupt_requested:
                                                continue
                                            state.is_playing = True
                                            state.speaker_rms = (
                                                compute_rms(part.inline_data.data) * 0.5
                                            )
                                            audio_b64 = base64.b64encode(
                                                part.inline_data.data
                                            ).decode("ascii")
                                            await ws.send_json(
                                                {"type": "audio", "data": audio_b64}
                                            )

                    except Exception as e:
                        print(f"\n[Gemini Receive Error: {e}]")
                        break

                print("\n[Gemini receive loop stopped.]")

            async def send_periodic_reminder():
                """Re-inject compact circuit summary every REMINDER_INTERVAL seconds."""
                while True:
                    await asyncio.sleep(REMINDER_INTERVAL)
                    reminder = [
                        types.Content(
                            role="user",
                            parts=[types.Part(text=reminder_text)],
                        ),
                    ]
                    await session.send_client_content(
                        turns=reminder, turn_complete=False
                    )
                    print(f"\n[Reminder: compact circuit analysis re-injected]")

            async def session_timeout():
                """Hard disconnect after SESSION_MAX_DURATION seconds."""
                await asyncio.sleep(SESSION_MAX_DURATION)
                print(f"\n[Session timeout: {SESSION_MAX_DURATION}s reached, disconnecting]")
                await ws.send_json({
                    "type": "session_expired",
                    "message": "Session expired after 20 minutes. Please reconnect.",
                })
                await ws.close(code=1000, reason="Session time limit reached")

            tasks = [
                asyncio.create_task(handle_client_messages()),
                asyncio.create_task(handle_gemini_responses()),
                asyncio.create_task(send_periodic_reminder()),
                asyncio.create_task(session_timeout()),
            ]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Session error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting ElectroTA server (manual VAD mode + function calling)...")
    print(
        f"Interrupt threshold: multiplier={INTERRUPT_MULTIPLIER}, min_rms={MIN_INTERRUPT_RMS}"
    )
    print(f"Quiet speech threshold: {QUIET_SPEECH_RMS}")
    print(
        f"Silence to end turn: {SILENCE_CHUNKS_TO_END} chunks ({SILENCE_CHUNKS_TO_END * 64}ms)"
    )
    print(f"Registered tool: probe()")
    print("Waiting for client connection on ws://0.0.0.0:8000/ws ...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=10 * 1024 * 1024)
