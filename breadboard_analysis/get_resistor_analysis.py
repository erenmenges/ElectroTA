import os
import re
import io
import mimetypes
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
# Import your custom direction finder script 
# (Assuming it is saved as 'resistor_direction_detect.py' in the same directory)
import resistor_direction_detect

# ── Tunable image enhancement values ──
SATURATION_FACTOR = 1.3   # 1.0 = original, >1.0 = more saturated
HIGHLIGHTS_FACTOR = 1.3    # 1.0 = original, >1.0 = brighter
SHARPNESS_FACTOR  = 1.3    # 1.0 = original, >1.0 = sharper

def enhance_image(image_bytes: bytes, mime_type: str) -> bytes:
    """
    Applies saturation, highlights (brightness), and sharpness adjustments
    to the image and returns the modified image bytes.
    """
    img = Image.open(io.BytesIO(image_bytes))

    img = ImageEnhance.Color(img).enhance(SATURATION_FACTOR)
    img = ImageEnhance.Brightness(img).enhance(HIGHLIGHTS_FACTOR)
    img = ImageEnhance.Sharpness(img).enhance(SHARPNESS_FACTOR)

    buf = io.BytesIO()
    fmt = "JPEG" if "jpeg" in mime_type or "jpg" in mime_type else "PNG"
    img.save(buf, format=fmt)
    return buf.getvalue()

def get_resistor_ohms(image_path: str) -> str:
    """
    Analyzes a resistor image using a local direction-finding script and the Gemini API 
    to return its resistance value in ohms.
    """
    # 1. Load the API key from the .env file
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    dotenv_path = parent_dir / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in .env file.")
    # 2. Call your resistor direction finder script
    direction = resistor_direction_detect.detect_reading_direction(image_path)
    
    if direction.startswith("error:"):
        return f"Error: {direction}"
    # 3. Read the image bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    # Determine the mime type based on file extension (e.g., image/jpeg or image/png)
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = 'image/jpeg' # Fallback

    # 3.5. Enhance the image before sending to Gemini
    image_bytes = enhance_image(image_bytes, mime_type)

    # 4. Construct the prompt with the inserted direction
    prompt = (
        f'How many ohms is this resistor? Start reading from the {direction}. '
        f'First look at the image and read the colors in order, starting from {direction}.'
        'Do not hallucinate or state colors you don\'t see.'
        'After you determine how many ohms, in the end, say "final_result=[your answer]" '
        'in which your answer is the numerical ohm result.'
    )
    # 5. Initialize the Gemini client and send the request
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt
        ],
        config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level='low')
  )
    )
    # 6. Parse out the final_result using a regular expression
    text = response.text
    
    # Looks for final_result=[...] or final_result=... and captures the value
    match = re.search(r'final_result=\[?([^\]\n]+)\]?', text, re.IGNORECASE)
    
    if match:
        # Return just the number
        return match.group(1).strip()
    else:
        # Fallback in case Gemini format varies slightly
        print("Raw response from Gemini:", text)
        return "Could not parse the final result from Gemini's response."
# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract resistor ohms from an image.")
    parser.add_argument("--image", required=True, help="Path to the resistor image")
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        ohms = get_resistor_ohms(args.image)
        print(f"\nFinal Extracted Value: {ohms} Ohms")
    else:
        print(f"Please place an image named '{args.image}' in the directory to test.")