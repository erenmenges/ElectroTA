import os
import sys
import asyncio
import base64
import json
import logging
import time
import threading
from collections import deque

import pyaudio
import traceback
import websockets
import serial
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
INPUT_AUDIO_RATE = 16000
OUTPUT_AUDIO_RATE = 24000
CHUNK_SIZE = 1024
SERVER_URL = os.getenv("SERVER_URL", "ws://localhost:8000/ws")

# --- Serial configuration ---
SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUD_RATE = 115200
PROBE_WINDOW_MS = 2000  # rolling average window

# Circuit diagram image is required as a command line argument
if len(sys.argv) < 2:
    print("Usage: python local_client.py <circuit_image_path> [serial_port]")
    print("Example: python local_client.py circuit.jpg COM4")
    sys.exit(1)

CIRCUIT_IMAGE_PATH = sys.argv[1]
if not os.path.isfile(CIRCUIT_IMAGE_PATH):
    print(f"Error: File not found: {CIRCUIT_IMAGE_PATH}")
    sys.exit(1)

if len(sys.argv) >= 3:
    SERIAL_PORT = sys.argv[2]


# --- Shared state for probe readings ---
# Each entry: (timestamp_seconds, voltage_float)
_probe_readings: deque = deque()
_probe_lock = threading.Lock()


def strip_trailing_padding(data):
    """
    ArduCAM's read_fifo_length() returns more bytes than the actual JPEG.
    After the real FF D9 end marker, the SPI reads return 0x00 padding.
    Find the last FF D9 and strip everything after it.
    """
    eoi_pos = data.rfind(b"\xff\xd9")
    if eoi_pos >= 0:
        trimmed = data[:eoi_pos + 2]
        padding_bytes = len(data) - len(trimmed)
        if padding_bytes > 0:
            print(f"  [debug] Stripped {padding_bytes} trailing padding bytes after FF D9")
        return trimmed
    return data


def validate_jpeg(data):
    """Check basic JPEG structure and print debug info."""
    has_start = data[:2] == b"\xff\xd8"
    has_end = data[-2:] == b"\xff\xd9"

    if not has_start:
        print("  [debug] WARNING: Missing JPEG start marker")
    if not has_end:
        print("  [debug] WARNING: Missing JPEG end marker")

    return has_start and has_end


_first_voltage_logged = False
_muted = False

def _record_voltage(voltage: float) -> None:
    """Store a voltage reading with its timestamp."""
    global _first_voltage_logged
    now = time.monotonic()
    with _probe_lock:
        _probe_readings.append((now, voltage))
        cutoff = now - 5.0
        while _probe_readings and _probe_readings[0][0] < cutoff:
            _probe_readings.popleft()
    if not _first_voltage_logged:
        _first_voltage_logged = True
        print(f"[Serial] First voltage reading received: {voltage:.2f} V")


def probe() -> dict:
    """Return the rolling average voltage over the last PROBE_WINDOW_MS."""
    now = time.monotonic()
    cutoff = now - (PROBE_WINDOW_MS / 1000.0)
    with _probe_lock:
        total_count = len(_probe_readings)
        recent = [(t, v) for t, v in _probe_readings if t >= cutoff]
    if not recent:
        print(f"  [probe debug] No readings in window. Total in deque: {total_count}")
        return {"voltage": None}
    avg = sum(v for _, v in recent) / len(recent)
    print(f"  [probe debug] {len(recent)} readings in window, avg={avg:.2f}V")
    return {"voltage": round(avg, 2)}


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

    # --- Open serial connection to ESP32 ---
    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Waiting for ESP32 to boot...")
    await asyncio.sleep(2.5)

    # Quick serial test: check if we can read anything at all
    waiting = ser.in_waiting
    print(f"[Serial test] Bytes in buffer after boot wait: {waiting}")
    if waiting > 0:
        sample = ser.read(min(waiting, 512))
        try:
            print(f"[Serial test] Sample data: {sample.decode('utf-8', errors='replace')!r}")
        except Exception:
            print(f"[Serial test] Sample raw bytes: {sample[:64]!r}")
    else:
        print("[Serial test] WARNING: No data received from ESP32 during boot.")
        print("[Serial test] Trying a blocking read (3s timeout)...")
        ser.timeout = 3
        sample = ser.read(128)
        ser.timeout = 1
        if sample:
            try:
                print(f"[Serial test] Got data: {sample.decode('utf-8', errors='replace')!r}")
            except Exception:
                print(f"[Serial test] Got raw bytes: {sample[:64]!r}")
        else:
            print("[Serial test] WARNING: Still nothing. ESP32 may not be sending data.")
            print("[Serial test] Possible causes:")
            print("[Serial test]   - ESP32 stuck in setup (SPI/camera check failed)")
            print("[Serial test]   - Baud rate mismatch (921600 may not work through USB dongle)")
            print("[Serial test]   - Wrong serial port or bad cable connection")

    ser.reset_input_buffer()
    ser.timeout = 0.05
    print("ESP32 serial ready.")

    # Queue for images received from the board
    image_queue: asyncio.Queue[bytes] = asyncio.Queue()

    print(f"Connecting to server at {SERVER_URL}...")
    try:
        async with websockets.connect(
            SERVER_URL,
            max_size=10 * 1024 * 1024,
            additional_headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            ping_interval=30,
            ping_timeout=120,
        ) as ws:
            print("Connected to server.")

            # Send the circuit diagram before anything else
            print(f"Sending circuit diagram: {CIRCUIT_IMAGE_PATH}")
            with open(CIRCUIT_IMAGE_PATH, "rb") as f:
                circuit_bytes = f.read()

            ext = os.path.splitext(CIRCUIT_IMAGE_PATH)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(ext, "image/jpeg")

            circuit_b64 = base64.b64encode(circuit_bytes).decode("ascii")
            await ws.send(
                json.dumps(
                    {
                        "type": "circuit_diagram",
                        "data": circuit_b64,
                        "mime_type": mime_type,
                    }
                )
            )
            print("Circuit diagram sent. Waiting for server to analyze...")

            while True:
                raw_msg = json.loads(await ws.recv())
                if raw_msg.get("type") == "status":
                    print(f"[Server]: {raw_msg['message']}")
                    if "circuit analyzed" in raw_msg["message"].lower():
                        break

            audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

            # ----------------------------------------------------------
            # Serial reader — runs in a thread, pushes images onto the
            # asyncio image_queue and records voltage readings.
            # ----------------------------------------------------------
            MAX_TEXT_BUFFER = 4096
            MAX_IMAGE_BUFFER = 2 * 1024 * 1024  # 2 MB

            def _serial_reader():
                """
                Blocking loop that reads the ESP32 serial stream byte-by-byte:
                  - Text lines containing "Probe Voltage:" are parsed and stored.
                  - JPEG data (FF D8 … FF D9) is collected and enqueued.
                Uses timeout-based reads (no in_waiting) for macOS compatibility.
                """
                text_buffer = bytearray()
                image_data = bytearray()
                receiving_image = False
                got_first_data = False
                line_count = 0

                while True:
                    try:
                        byte = ser.read(1)
                        if not byte:
                            continue

                        if not got_first_data:
                            got_first_data = True
                            print("[Serial] First data arriving from ESP32")

                        if not receiving_image:
                            text_buffer += byte

                            if len(text_buffer) > MAX_TEXT_BUFFER:
                                logging.warning("Text buffer exceeded limit, discarding")
                                text_buffer = bytearray()
                                continue

                            # Detect JPEG start marker (FF D8)
                            if len(text_buffer) >= 2 and text_buffer[-2:] == b"\xff\xd8":
                                receiving_image = True
                                image_data = bytearray(b"\xff\xd8")
                                text_buffer = bytearray()
                                print("[Serial] JPEG start marker detected")

                            # Detect end of a text line
                            elif text_buffer[-1:] == b"\n":
                                try:
                                    line = text_buffer.decode("utf-8").strip()
                                except UnicodeDecodeError:
                                    print(f"[Serial] Could not decode line bytes: {text_buffer[:50]!r}")
                                    text_buffer = bytearray()
                                    continue

                                line_count += 1
                                if line_count <= 10:
                                    print(f"[Serial] Line {line_count}: {line!r}")

                                if "Probe Voltage:" in line:
                                    try:
                                        raw = line.split(":")[-1].strip()
                                        voltage_str = raw.replace("V", "").strip()
                                        voltage = float(voltage_str)
                                        _record_voltage(voltage)
                                    except ValueError as e:
                                        print(f"[Serial] Failed to parse voltage from {line!r}: {e}")
                                text_buffer = bytearray()
                        else:
                            # Accumulating JPEG bytes
                            image_data += byte

                            if len(image_data) > MAX_IMAGE_BUFFER:
                                logging.warning("Image buffer exceeded limit, discarding")
                                image_data = bytearray()
                                receiving_image = False
                                continue

                            # Detect JPEG end marker (FF D9)
                            if len(image_data) >= 2 and image_data[-2:] == b"\xff\xd9":
                                image_data = strip_trailing_padding(image_data)
                                print(f"[Serial] JPEG complete: {len(image_data)} bytes")
                                if validate_jpeg(image_data):
                                    image_queue.put_nowait(bytes(image_data))
                                else:
                                    print("[Serial] Skipping — invalid JPEG")
                                image_data = bytearray()
                                receiving_image = False
                    except Exception as e:
                        logging.error(f"Serial reader error: {e}")
                        time.sleep(0.1)

            # Start the serial reader in a daemon thread
            serial_thread = threading.Thread(target=_serial_reader, daemon=True)
            serial_thread.start()
            print("Serial reader started. Photos will be sent automatically on capture.")

            # ----------------------------------------------------------
            # Async tasks
            # ----------------------------------------------------------
            async def send_audio():
                """Continuously read mic and send raw audio to server."""
                global _muted
                print("Microphone active. You can start speaking...")
                print("  (type 'm' + Enter to mute/unmute)")
                while True:
                    audio_data = await asyncio.to_thread(
                        mic_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    if not _muted:
                        audio_b64 = base64.b64encode(audio_data).decode("ascii")
                        await ws.send(json.dumps({"type": "audio", "data": audio_b64}))

            async def mute_toggle_listener():
                """Read stdin lines; toggle mute when user types 'm'."""
                global _muted
                while True:
                    line = await asyncio.to_thread(sys.stdin.readline)
                    if line.strip().lower() == "m":
                        _muted = not _muted
                        state = "MUTED" if _muted else "UNMUTED"
                        print(f"[Mic {state}]")

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
                    elif msg_type == "probe_request":
                        print("\n[Server requested probe reading]")
                        result = probe()
                        print(f"  probe() = {result}")
                        await ws.send(
                            json.dumps({"type": "probe_result", "data": result})
                        )

            async def show_progress_bar(duration: float, label: str = "Analyzing"):
                """Print a time-based progress bar that fills over `duration` seconds."""
                bar_width = 30
                start = time.monotonic()
                while True:
                    elapsed = time.monotonic() - start
                    fraction = min(elapsed / duration, 1.0)
                    filled = int(bar_width * fraction)
                    bar = "#" * filled + "-" * (bar_width - filled)
                    percent = int(fraction * 100)
                    print(f"\r{label}: [{bar}] {percent}%", end="", flush=True)
                    if fraction >= 1.0:
                        print()
                        break
                    await asyncio.sleep(0.25)

            async def image_sender():
                """Wait for images from the serial reader and forward to server."""
                desktop_path = os.path.expanduser("~/Desktop")
                image_counter = 0
                while True:
                    jpeg_bytes = await image_queue.get()

                    image_counter += 1
                    filename = f"esp32_photo_{image_counter}.jpg"
                    save_path = os.path.join(desktop_path, filename)
                    with open(save_path, "wb") as img_file:
                        img_file.write(jpeg_bytes)
                    print(f"[>>> Photo saved to {save_path}]")

                    photo_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                    print(f"\n[>>> Sending photo from board: {len(jpeg_bytes)} bytes]")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "image",
                                "data": photo_b64,
                                "mime_type": "image/jpeg",
                            }
                        )
                    )
                    asyncio.create_task(show_progress_bar(40.0))

            await asyncio.gather(
                send_audio(),
                play_audio(),
                receive_messages(),
                image_sender(),
                mute_toggle_listener(),
            )
    finally:
        try:
            ser.close()
        except Exception:
            pass
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