"""
Receives photos from ESP32 over serial and runs breadboard analysis on each one.
Usage: python esp32_analyzer.py [--no-analysis] [serial_port]
"""

import sys
import os
import time
import logging
import threading
import argparse

import serial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "breadboard_analysis"))
from complete_analysis import run_analysis

logging.basicConfig(level=logging.WARNING)

SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUD_RATE = 115200
MAX_IMAGE_BUFFER = 2 * 1024 * 1024
READ_CHUNK_SIZE = 4096


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
    print(f"  [debug] Total bytes: {len(data)}")
    print(f"  [debug] First 16 bytes: {data[:16].hex(' ')}")
    print(f"  [debug] Last  16 bytes: {data[-16:].hex(' ')}")

    has_start = data[:2] == b"\xff\xd8"
    has_end = data[-2:] == b"\xff\xd9"
    print(f"  [debug] Starts with FF D8: {has_start}")
    print(f"  [debug] Ends with FF D9:   {has_end}")

    if not has_start:
        print("  [debug] WARNING: Missing JPEG start marker")
    if not has_end:
        print("  [debug] WARNING: Missing JPEG end marker")

    return has_start and has_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-analysis", action="store_true", help="Skip analysis, only receive and save photos")
    parser.add_argument("--no-gemini-analysis", action="store_true", help="Run component detection and ohm readings, but skip the final Gemini API call")
    parser.add_argument("port", nargs="?", default=SERIAL_PORT, help="Serial port")
    args = parser.parse_args()

    port = args.port
    skip_analysis = args.no_analysis
    skip_gemini_analysis = args.no_gemini_analysis

    print(f"Opening serial port {port} at {BAUD_RATE} baud...")
    if skip_analysis:
        print("Analysis DISABLED (--no-analysis)")
    elif skip_gemini_analysis:
        print("Gemini final analysis DISABLED (--no-gemini-analysis)")
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    print("Waiting for ESP32 to boot...")
    time.sleep(2.5)
    ser.reset_input_buffer()
    ser.timeout = 0.1
    print("ESP32 serial ready. Waiting for photos...\n")

    text_buffer = bytearray()
    image_data = bytearray()
    receiving_image = False
    expected_image_size = 0
    image_count = 0
    receive_start_time = 0
    last_progress = 0

    while True:
        try:
            if not receiving_image:
                byte = ser.read(1)
                if not byte:
                    continue

                text_buffer += byte

                if len(text_buffer) > 4096:
                    text_buffer = bytearray()
                    continue

                if len(text_buffer) >= 2 and text_buffer[-2:] == b"\xff\xd8":
                    receiving_image = True
                    image_data = bytearray(b"\xff\xd8")
                    receive_start_time = time.time()
                    last_progress = 0
                    text_before = text_buffer[:-2]
                    text_buffer = bytearray()
                    print(f"[Serial] JPEG start marker detected (photo #{image_count + 1})")
                    if expected_image_size > 0:
                        print(f"[Serial] Expecting {expected_image_size} bytes")
                    if text_before:
                        print(f"  [debug] {len(text_before)} bytes between size line and JPEG start: {text_before[:40]}")

                elif text_buffer[-1:] == b"\n":
                    try:
                        line = text_buffer.decode("utf-8").strip()
                        if line and "Probe Voltage" not in line:
                            print(f"[Serial] {line}")
                        if line.startswith("Image size:") and "bytes" in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "size:" and i + 1 < len(parts):
                                    expected_image_size = int(parts[i + 1])
                                    break
                    except (UnicodeDecodeError, ValueError):
                        pass
                    text_buffer = bytearray()
            else:
                if expected_image_size > 0:
                    remaining = expected_image_size - len(image_data)
                    to_read = min(remaining, READ_CHUNK_SIZE)
                else:
                    to_read = max(1, ser.in_waiting) if ser.in_waiting > 0 else 1

                chunk = ser.read(to_read)
                if not chunk:
                    continue

                image_data += chunk

                received = len(image_data)
                if received - last_progress >= 10000:
                    elapsed = time.time() - receive_start_time
                    rate = received / elapsed if elapsed > 0 else 0
                    if expected_image_size > 0:
                        pct = (received / expected_image_size) * 100
                        print(f"  [debug] Received {received}/{expected_image_size} bytes ({pct:.0f}%) — {rate:.0f} B/s — in_waiting: {ser.in_waiting}")
                    else:
                        print(f"  [debug] Received {received} bytes — {rate:.0f} B/s — in_waiting: {ser.in_waiting}")
                    last_progress = received

                if len(image_data) > MAX_IMAGE_BUFFER:
                    print("[Serial] Image too large, discarding")
                    image_data = bytearray()
                    receiving_image = False
                    expected_image_size = 0
                    continue

                image_done = False
                if expected_image_size > 0 and len(image_data) >= expected_image_size:
                    image_done = True
                elif expected_image_size == 0 and len(image_data) >= 2 and image_data[-2:] == b"\xff\xd9":
                    image_done = True

                if image_done:
                    elapsed = time.time() - receive_start_time
                    if expected_image_size > 0:
                        image_data = image_data[:expected_image_size]
                    raw_size = len(image_data)
                    image_data = strip_trailing_padding(image_data)
                    image_count += 1
                    print(f"[Serial] JPEG complete: {len(image_data)} bytes (raw {raw_size}) (photo #{image_count}) in {elapsed:.2f}s")

                    print(f"  [debug] Validating JPEG data...")
                    valid = validate_jpeg(image_data)

                    desktop_path = os.path.expanduser("~/Desktop")
                    save_path = os.path.join(desktop_path, f"esp32_photo_{image_count}.jpg")
                    with open(save_path, "wb") as f:
                        f.write(image_data)
                    print(f"[Serial] Saved to {save_path}")

                    image_data = bytearray()
                    receiving_image = False
                    expected_image_size = 0

                    if not valid:
                        print("[Serial] Skipping analysis — invalid JPEG\n")
                    elif not skip_analysis:
                        photo_num = image_count
                        photo_path = save_path

                        def analyze(num, path):
                            print(f"\n=== Running analysis on photo #{num} ===")
                            try:
                                result = run_analysis(path, skip_gemini_analysis=skip_gemini_analysis)
                                print(f"\n--- Analysis Result (photo #{num}) ---")
                                print(result)
                                print(f"--- End of Result (photo #{num}) ---\n")
                            except Exception as e:
                                print(f"Analysis failed (photo #{num}): {e}")

                        thread = threading.Thread(target=analyze, args=(photo_num, photo_path))
                        thread.daemon = True
                        thread.start()
                    else:
                        print("[Serial] Skipping analysis (--no-analysis)\n")

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            logging.error(f"Serial read error: {e}")
            time.sleep(0.1)

    ser.close()
    print("Serial port closed.")


if __name__ == "__main__":
    main()
