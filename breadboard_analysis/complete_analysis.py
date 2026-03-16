"""
Complete breadboard analysis pipeline.

Usage (CLI):
    python complete_analysis.py path/to/image.jpg
    python complete_analysis.py path/to/image.jpg --crops-dir /custom/crops

Usage (import):
    from complete_analysis import run_analysis
    result = run_analysis("path/to/image.jpg")
"""

import sys
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

import detect_components


def _get_mime_type(file_path: str) -> str | None:
    """Return MIME type for image files, None for non-image files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return None


def run_analysis(image_path: str, crops_dir: str = None, skip_gemini_analysis: bool = False) -> str:
    """
    Run the complete breadboard analysis pipeline.

    1. Runs detect_components on the image (produces crops).
    2. If not skip_gemini_analysis: collects files, sends to Gemini, returns response.
    3. If skip_gemini_analysis: skips the final Gemini call, returns a summary.

    Args:
        image_path: Path to the input breadboard image.
        crops_dir: Directory for crop outputs.
                   Defaults to <project_root>/crops.
        skip_gemini_analysis: If True, do not call Gemini for final analysis.

    Returns:
        The Gemini API response text, or a summary if skip_gemini_analysis.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if crops_dir is None:
        crops_dir = str(project_root / "crops")

    prompt_path = str(script_dir / "prompt.txt")

    # Load .env from project root
    load_dotenv(dotenv_path=project_root / ".env")

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(
            "API_KEY not found. Set it in .env or as an environment variable."
        )

    # --- Step 1: Run component detection ---
    # detect_components.main() derives its crops_dir from output_path:
    #   crops_dir = os.path.join(os.path.dirname(output_path) or ".", "crops")
    # So we set output_path so that dirname(output_path)/crops == our crops_dir.
    crops_parent = str(Path(crops_dir).parent)
    output_path = os.path.join(crops_parent, "segmented_output.png")

    print(f"Running component detection on: {image_path}")
    detect_components.main(image_path, output_path=output_path)
    print("Component detection complete. Results saved to:", crops_dir)

    if skip_gemini_analysis:
        print("Skipping Gemini analysis (--no-gemini-analysis).")
        return "Component detection and ohm readings done. See crops/ for results."

    # --- Step 2: Read prompt text ---
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    if not prompt_text:
        raise RuntimeError(f"Prompt file is empty: {prompt_path}")

    # --- Step 3: Collect all files from crops/ ---
    contents = [prompt_text]

    crop_files = sorted(os.listdir(crops_dir))
    for filename in crop_files:
        if filename.startswith("."):
            continue

        file_path = os.path.join(crops_dir, filename)
        if not os.path.isfile(file_path):
            continue

        mime = _get_mime_type(file_path)
        if mime is not None:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime)
            )
            print(f"  Added image: {filename}")
        elif filename.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_text = f.read()
            contents.append(f"Contents of {filename}:\n{json_text}")
            print(f"  Added text file: {filename}")

    # --- Step 4: Send to Gemini ---
    print("Sending to Gemini (gemini-3-flash-preview, thinking=medium) ...")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
        ),
    )

    result_text = response.text
    print("Gemini response received.")
    return result_text


def main():
    parser = argparse.ArgumentParser(
        description="Run complete breadboard analysis pipeline."
    )
    parser.add_argument("image_path", help="Path to the input breadboard image")
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Directory for crop outputs (default: <project_root>/crops)",
    )
    parser.add_argument(
        "--no-gemini-analysis",
        action="store_true",
        help="Run component detection and ohm readings, skip final Gemini API call",
    )
    args = parser.parse_args()

    result = run_analysis(
        args.image_path,
        crops_dir=args.crops_dir,
        skip_gemini_analysis=args.no_gemini_analysis,
    )
    print("\n--- Gemini Analysis Result ---")
    print(result)


if __name__ == "__main__":
    main()
