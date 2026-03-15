import os
import sys
import asyncio
import base64
import json
import logging
import pyaudio
import traceback
import websockets
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# Platform-specific single-keypress reader
if os.name == "nt":
    import msvcrt

    def _blocking_get_key():
        return msvcrt.getwch()
else:
    import tty
    import termios

    def _blocking_get_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
INPUT_AUDIO_RATE = 16000
OUTPUT_AUDIO_RATE = 24000
CHUNK_SIZE = 1024

SERVER_URL = "ws://localhost:8000/ws"

# Circuit diagram image is required as a command line argument
if len(sys.argv) < 2:
    print("Usage: python local_client.py <circuit_image_path>")
    print("Example: python local_client.py circuit.jpg")
    sys.exit(1)

CIRCUIT_IMAGE_PATH = sys.argv[1]
if not os.path.isfile(CIRCUIT_IMAGE_PATH):
    print(f"Error: File not found: {CIRCUIT_IMAGE_PATH}")
    sys.exit(1)

PHOTO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")


def probe():
    """Read probe measurements. Replace with real hardware calls."""
    return {"voltage": 2.0, "current": 5.0}


async def main():
    p = pyaudio.PyAudio()
    mic_stream = p.open(
        format=AUDIO_FORMAT,
        channels=AUDIO_CHANNELS,
        rate=INPUT_AUDIO_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    speaker_stream = p.open(
        format=AUDIO_FORMAT,
        channels=AUDIO_CHANNELS,
        rate=OUTPUT_AUDIO_RATE,
        output=True,
    )

    print(f"Connecting to server at {SERVER_URL}...")

    try:
        async with websockets.connect(
            SERVER_URL,
            max_size=10 * 1024 * 1024,
            additional_headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        ) as ws:
            print("Connected to server.")

            # Send the circuit diagram before anything else
            print(f"Sending circuit diagram: {CIRCUIT_IMAGE_PATH}")
            with open(CIRCUIT_IMAGE_PATH, "rb") as f:
                circuit_bytes = f.read()

            # Determine mime type from file extension
            ext = os.path.splitext(CIRCUIT_IMAGE_PATH)[1].lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/jpeg")

            circuit_b64 = base64.b64encode(circuit_bytes).decode("ascii")
            await ws.send(json.dumps({
                "type": "circuit_diagram",
                "data": circuit_b64,
                "mime_type": mime_type,
            }))
            print("Circuit diagram sent. Waiting for server to analyze...")

            # Wait for the server to confirm the circuit was analyzed
            while True:
                raw_msg = json.loads(await ws.recv())
                if raw_msg.get("type") == "status":
                    print(f"[Server]: {raw_msg['message']}")
                    if "circuit analyzed" in raw_msg["message"].lower():
                        break

            audio_queue = asyncio.Queue()

            async def send_audio():
                """Continuously read mic and send raw audio to server."""
                print("Microphone active. You can start speaking...")
                while True:
                    audio_data = await asyncio.to_thread(
                        mic_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    audio_b64 = base64.b64encode(audio_data).decode("ascii")
                    await ws.send(json.dumps({"type": "audio", "data": audio_b64}))

            async def play_audio():
                """Read from the audio queue and play through the speaker."""
                while True:
                    audio_data = await audio_queue.get()
                    await asyncio.to_thread(speaker_stream.write, audio_data)

            async def receive_messages():
                """Receive messages from server and handle them."""
                async for raw in ws:
                    msg = json.loads(raw)
                    msg_type = msg.get("type")

                    if msg_type == "audio":
                        audio_data = base64.b64decode(msg["data"])
                        await audio_queue.put(audio_data)

                    elif msg_type == "interrupt":
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        print("\n[Interrupt: cleared audio buffer]")

                    elif msg_type == "status":
                        print(f"[Server]: {msg['message']}")

                    elif msg_type == "transcript_output":
                        print(f"{msg['text']}", end="", flush=True)

                    elif msg_type == "transcript_input":
                        print(f"\n  [You]: {msg['text']}", end="", flush=True)

                    elif msg_type == "probe_request":
                        print("\n[Server requested probe reading]")
                        result = probe()
                        print(f"  probe() = {result}")
                        await ws.send(
                            json.dumps({"type": "probe_result", "data": result})
                        )

            async def keypress_listener():
                """Listen for keypresses: 'p' sends photo, 'q' quits."""
                print(f"Press 'p' to send photo: {PHOTO_PATH}")
                print("Press 'q' to quit.\n")

                while True:
                    key = await asyncio.to_thread(_blocking_get_key)

                    if key == "q":
                        print("\n[Quit requested]")
                        os._exit(0)

                    if key != "p":
                        continue

                    if not os.path.isfile(PHOTO_PATH):
                        print(f"\n[Photo not found: {PHOTO_PATH}]")
                        continue

                    with open(PHOTO_PATH, "rb") as f:
                        photo_bytes = f.read()

                    photo_b64 = base64.b64encode(photo_bytes).decode("ascii")
                    print(f"\n[>>> Sending photo: {len(photo_bytes)} bytes]")

                    await ws.send(
                        json.dumps(
                            {
                                "type": "image",
                                "data": photo_b64,
                                "mime_type": "image/jpeg",
                            }
                        )
                    )
                    print("[>>> Photo sent to server]")

            await asyncio.gather(
                send_audio(),
                play_audio(),
                receive_messages(),
                keypress_listener(),
            )

    finally:
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass
        try:
            speaker_stream.stop_stream()
            speaker_stream.close()
        except Exception:
            pass
        p.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print("\n=== CONNECTION ERROR ===")
        print(f"Error Type: {type(e).__name__}")
        print(f"Message: {e}")
        print("\nFull Traceback:")
        traceback.print_exc()
        print("========================\n")
