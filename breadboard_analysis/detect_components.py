""""
Segment capacitors, diodes, and resistors in an image using Gemini 2.5.
Usage:
    pip install google-genai Pillow numpy
    export GOOGLE_API_KEY="your-api-key"
    python segment_components.py path/to/image.jpg
"""
import sys
import os
import json
import base64
import io
import re
import tempfile
import time
import threading
import dataclasses
import shutil
import argparse
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from google import genai
from google.genai import types
import get_resistor_analysis
import diode_cathode_detector
import capacitor_cathode_detection
import breadboard_detection
from pathlib import Path
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_json(raw: str) -> str:
    """Strip markdown JSON fences if present."""
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            raw = "\n".join(lines[i + 1:])
            raw = raw.split("```")[0]
            break
    return raw
@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    y0: int
    x0: int
    y1: int
    x1: int
    mask: np.ndarray   # full-image sized, values 0-255
    label: str
def parse_segmentation_masks(
    predicted_str: str, *, img_height: int, img_width: int
) -> list[SegmentationMask]:
    cleaned = parse_json(predicted_str)
    # Debug: dump what the model actually returned
    print("\n--- Raw model response (first 2000 chars) ---")
    print(predicted_str[:2000])
    print("--- end ---\n")
    items = json.loads(cleaned)
    masks: list[SegmentationMask] = []
    for item in items:
        # Flexible key lookup for the bounding box
        box = item.get("box_2d") or item.get("bounding_box") or item.get("bbox")
        if box is None:
            print(f"  [skip] no bounding box found in item: {list(item.keys())}")
            continue
        abs_y0 = int(box[0] / 1000 * img_height)
        abs_x0 = int(box[1] / 1000 * img_width)
        abs_y1 = int(box[2] / 1000 * img_height)
        abs_x1 = int(box[3] / 1000 * img_width)
        if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
            print(f"  [skip] invalid box for '{item.get('label', '?')}': {box}")
            continue
        bbox_h = abs_y1 - abs_y0
        bbox_w = abs_x1 - abs_x0
        if bbox_h < 1 or bbox_w < 1:
            continue
        label = item.get("label", "")
        # --- Try to decode the segmentation mask --------------------------
        png_str: str | None = item.get("mask")
        if png_str and png_str.startswith("data:image/png;base64,"):
            try:
                png_bytes = base64.b64decode(
                    png_str.removeprefix("data:image/png;base64,")
                )
                mask_img = Image.open(io.BytesIO(png_bytes))
                mask_img = mask_img.resize((bbox_w, bbox_h), Image.Resampling.BILINEAR)
                full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = np.array(mask_img)
            except (OSError, ValueError, SyntaxError) as e:
                print(f"  [warn] broken mask for '{label}', using bounding box fill: {e}")
                full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = 255
        else:
            # No valid mask — fall back to a solid bounding-box fill
            print(f"  [info] no segmentation mask for '{label}'; using bounding box fill")
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = 255
        masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, full_mask, label))
    return masks
# ---------------------------------------------------------------------------
# Numbered-label assignment
# ---------------------------------------------------------------------------
_KNOWN_TYPES = ["capacitor", "diode", "led", "resistor"]
def _extract_component_type(label: str) -> str:
    """Extract the base component type from a Gemini label.
    Looks for known component keywords; returns the first match.
    Falls back to the full (lowered, stripped) label.
    """
    lower = label.lower()
    for comp_type in _KNOWN_TYPES:
        if comp_type in lower:
            return comp_type
    return label.strip()
def assign_numbered_labels(masks: list[SegmentationMask]) -> list[SegmentationMask]:
    """Replace each mask's label with a numbered ID like LED1, diode2, etc.
    Counts each component type independently so numbering starts at 1
    per type.
    """
    type_counts: dict[str, int] = {}
    numbered: list[SegmentationMask] = []
    for m in masks:
        comp_type = _extract_component_type(m.label)
        count = type_counts.get(comp_type, 0) + 1
        type_counts[comp_type] = count
        numbered_label = f"{comp_type}{count}"
        numbered.append(
            SegmentationMask(m.y0, m.x0, m.y1, m.x1, m.mask, numbered_label)
        )
    return numbered
# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def overlay_mask(img: Image.Image, mask: np.ndarray, color_name: str, alpha: float = 0.5) -> Image.Image:
    rgb = ImageColor.getrgb(color_name)
    rgba = rgb + (int(alpha * 255),)
    img_rgba = img.convert("RGBA")
    w, h = img_rgba.size
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[mask > 127] = rgba
    overlay = Image.fromarray(layer, "RGBA")
    return Image.alpha_composite(img_rgba, overlay)
COLORS = [
    "red", "green", "blue", "yellow", "orange", "cyan",
    "magenta", "lime", "purple", "teal", "coral", "gold",
]
def draw_results(img: Image.Image, masks: list[SegmentationMask]) -> Image.Image:
    # overlay masks
    for i, m in enumerate(masks):
        img = overlay_mask(img, m.mask, COLORS[i % len(COLORS)])
    draw = ImageDraw.Draw(img)
    # try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    # bounding boxes
    for i, m in enumerate(masks):
        color = COLORS[i % len(COLORS)]
        draw.rectangle([(m.x0, m.y0), (m.x1, m.y1)], outline=color, width=3)
    # labels
    for i, m in enumerate(masks):
        color = COLORS[i % len(COLORS)]
        if m.label:
            draw.text((m.x0 + 4, m.y0 - 18), m.label, fill=color, font=font)
    return img
def sanitize_label(label: str) -> str:
    """Turn a label into a safe filename fragment (lowercase, underscores)."""
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "unknown"
# ---------------------------------------------------------------------------
# Spatial-relationship helpers
# ---------------------------------------------------------------------------
def _center(m: SegmentationMask) -> tuple[float, float]:
    """Return (cx, cy) of a mask's bounding box."""
    return (m.x0 + m.x1) / 2.0, (m.y0 + m.y1) / 2.0
def _relative_position(source: SegmentationMask, target: SegmentationMask) -> str:
    """Describe where *source* sits relative to *target*
    (e.g. 'below-left of').  Uses centre-to-centre vector."""
    sx, sy = _center(source)
    tx, ty = _center(target)
    dx = sx - tx          # positive → source is to the right
    dy = sy - ty          # positive → source is below  (image coords)
    # If centres are within 40 % of the larger bbox span on an axis
    # we consider them roughly aligned on that axis.
    span_x = max(source.x1 - source.x0, target.x1 - target.x0, 1)
    span_y = max(source.y1 - source.y0, target.y1 - target.y0, 1)
    near_x = abs(dx) < span_x * 0.40
    near_y = abs(dy) < span_y * 0.40
    vert = ""
    if not near_y:
        vert = "above" if dy < 0 else "below"
    horiz = ""
    if not near_x:
        horiz = "left" if dx < 0 else "right"
    if vert and horiz:
        return f"{vert}-{horiz} of"
    if vert:
        return f"directly {vert}"
    if horiz:
        return f"to the {horiz} of"
    return "overlapping / very close to"
def build_relationship_lines(
    idx: int, masks: list[SegmentationMask]
) -> list[str]:
    """Human-readable lines describing masks[idx] position vs every other."""
    source = masks[idx]
    lines: list[str] = []
    for j, target in enumerate(masks):
        if j == idx:
            continue
        rel = _relative_position(source, target)
        lines.append(f"  {rel}  {target.label}")
    return lines
# ---------------------------------------------------------------------------
# Annotated crop extraction
# ---------------------------------------------------------------------------
MIN_CROP_WIDTH  = 480       # minimum width after upscale (px)
CROP_PADDING     = 0        # px of context added around each bbox crop
def analyze_components(
    img: Image.Image,
    masks: list[SegmentationMask],
    output_dir: str,
    img_fullres: Image.Image | None = None,
    save_temps: bool = False,
) -> None:
    """Run per-component analysis (resistor ohms, diode/capacitor cathode
    orientation) and save annotations.json to *output_dir*.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_annotations: list[dict] = []

    if img_fullres is not None:
        full_w, full_h = img_fullres.size
        thumb_w, thumb_h = img.size
        scale_x = full_w / thumb_w
        scale_y = full_h / thumb_h
    else:
        img_fullres = img
        scale_x = 1.0
        scale_y = 1.0

    for i, m in enumerate(masks):
        # --- Crop from the full-resolution image for analysis -------------
        pad = 5 if "diode" in m.label.lower() or "capacitor" in m.label.lower() else CROP_PADDING
        fr_w, fr_h = img_fullres.size
        fr_pad_x0 = max(int(m.x0 * scale_x) - pad, 0)
        fr_pad_y0 = max(int(m.y0 * scale_y) - pad, 0)
        fr_pad_x1 = min(int(m.x1 * scale_x) + pad, fr_w)
        fr_pad_y1 = min(int(m.y1 * scale_y) + pad, fr_h)
        analysis_crop = img_fullres.crop(
            (fr_pad_x0, fr_pad_y0, fr_pad_x1, fr_pad_y1)
        ).convert("RGB")

        # --- Resistor analysis: call get_resistor_ohms --------------------
        ohms_value: int | None = None
        if "resistor" in m.label.lower():
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(tmp_fd)
                tmp_fd = None
                resistor_crop_for_analysis = analysis_crop
                short_side = min(analysis_crop.width, analysis_crop.height)
                if short_side < MIN_CROP_WIDTH:
                    upscale = MIN_CROP_WIDTH / short_side
                    resistor_crop_for_analysis = analysis_crop.resize(
                        (int(analysis_crop.width * upscale),
                         int(analysis_crop.height * upscale)),
                        Image.Resampling.LANCZOS,
                    )
                resistor_crop_for_analysis.save(tmp_path, "JPEG", quality=95)
                raw_result = get_resistor_analysis.get_resistor_ohms(tmp_path)
                ohms_value = int(raw_result)
                print(f"  Resistor analysis for '{m.label}': {ohms_value} Ω")
            except (ValueError, TypeError):
                print(
                    f"  [warn] resistor analysis for '{m.label}' returned "
                    f"non-integer result: {raw_result!r}; skipping ohms embed"
                )
                ohms_value = None
            except Exception as e:
                print(
                    f"  [warn] resistor analysis failed for '{m.label}': "
                    f"{type(e).__name__}: {e}; skipping ohms embed"
                )
                ohms_value = None
            finally:
                if save_temps and tmp_path and os.path.exists(tmp_path):
                    save_name = f"{sanitize_label(m.label)}_temp.jpg"
                    save_dest = os.path.join(output_dir, save_name)
                    shutil.copy2(tmp_path, save_dest)
                    print(f"  Saved temp crop -> {save_dest}")
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        # --- Diode cathode detection (not for LEDs) -----------------------
        diode_cathode_dir: str | None = None
        if "diode" in m.label.lower() and "led" not in m.label.lower():
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(tmp_fd)
                tmp_fd = None
                analysis_crop.save(tmp_path, "JPEG", quality=95)
                diode_cathode_dir = diode_cathode_detector.detect_cathode_direction(tmp_path)
                print(f"  Diode cathode detection for '{m.label}': {diode_cathode_dir}")
            except Exception as e:
                print(
                    f"  [warn] diode cathode detection failed for '{m.label}': "
                    f"{type(e).__name__}: {e}; skipping cathode embed"
                )
                diode_cathode_dir = None
            finally:
                if save_temps and tmp_path and os.path.exists(tmp_path):
                    save_name = f"{sanitize_label(m.label)}_temp.jpg"
                    save_dest = os.path.join(output_dir, save_name)
                    shutil.copy2(tmp_path, save_dest)
                    print(f"  Saved temp crop -> {save_dest}")
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        # --- Capacitor cathode detection ----------------------------------
        capacitor_cathode_dir: str | None = None
        if "capacitor" in m.label.lower():
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(tmp_fd)
                tmp_fd = None
                analysis_crop.save(tmp_path, "JPEG", quality=95)
                capacitor_cathode_dir = capacitor_cathode_detection.detect_cathode_direction(tmp_path)
                print(f"  Capacitor cathode detection for '{m.label}': {capacitor_cathode_dir}")
            except Exception as e:
                print(
                    f"  [warn] capacitor cathode detection failed for '{m.label}': "
                    f"{type(e).__name__}: {e}; skipping cathode embed"
                )
                capacitor_cathode_dir = None
            finally:
                if save_temps and tmp_path and os.path.exists(tmp_path):
                    save_name = f"{sanitize_label(m.label)}_temp.jpg"
                    save_dest = os.path.join(output_dir, save_name)
                    shutil.copy2(tmp_path, save_dest)
                    print(f"  Saved temp crop -> {save_dest}")
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        # --- Collect annotation data --------------------------------------
        rel_lines = build_relationship_lines(i, masks)
        annotation: dict = {
            "label": m.label,
            "bounding_box": {
                "x0": m.x0, "y0": m.y0,
                "x1": m.x1, "y1": m.y1,
                "width": m.x1 - m.x0,
                "height": m.y1 - m.y0,
            },
        }
        if ohms_value is not None:
            annotation["resistance_ohms"] = ohms_value
        if diode_cathode_dir is not None:
            annotation["cathode_direction"] = diode_cathode_dir
        if capacitor_cathode_dir is not None:
            annotation["cathode_direction"] = capacitor_cathode_dir
        if rel_lines:
            annotation["spatial_relationships"] = [
                rl.strip() for rl in rel_lines
            ]
        all_annotations.append(annotation)
    # --- Save all annotations as a single JSON file -----------------------
    json_path = os.path.join(output_dir, "annotations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)
    print(f"  Saved annotations -> {json_path}")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PROMPT = (
    "Give the segmentation masks for all capacitors, diodes, transistors, and resistors in this image. "
    "CRUCIAL: The image has UNMISTAKABLY more than one component that you need to identify."
    "Output a JSON list of bounding boxes where each entry contains "
    'the 2D bounding box in the key "box_2d", '
    'and the text label in the key "label". '
    "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000. "
    "When creating bounding boxes of resistors, only bound the full body. Do not include the metal wirings."
    "Use descriptive labels that include the component type (capacitor, diode, LED, transistor, or resistor)."
)
def main(image_path: str, output_path: str = "segmented_output.png", save_temps: bool = False) -> None:
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    dotenv_path = parent_dir / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("API_KEY")
    if not api_key:
        sys.exit("Error: set the GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)
    # Load the full-resolution original (kept for high-quality analysis crops)
    im_fullres = Image.open(image_path)
    # Create a thumbnail copy for Gemini segmentation (works best ≤1024px)
    im = im_fullres.copy()
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    width, height = im.size
    print(f"Image size after resize: {width} x {height}", flush=True)
    model_name = "gemini-3-flash-preview"
    print(f"Calling Gemini ({model_name}) for segmentation …")
    print(f"  Timeout: 120s | Thinking level: minimal", flush=True)
    t0 = time.time()
    # Heartbeat thread: prints every 10s so we know the process isn't dead
    heartbeat_stop = threading.Event()
    def _heartbeat():
        while not heartbeat_stop.is_set():
            heartbeat_stop.wait(10)
            if not heartbeat_stop.is_set():
                elapsed = time.time() - t0
                print(f"  … still waiting for Gemini ({elapsed:.0f}s elapsed)", flush=True)
    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    response = None
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[PROMPT, im],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level='minimal'),
                http_options=types.HttpOptions(timeout=120_000),
            ),
        )
    except Exception as e:
        heartbeat_stop.set()
        elapsed = time.time() - t0
        print(f"\n--- Gemini API call FAILED after {elapsed:.1f}s ---")
        print(f"  Exception type : {type(e).__name__}")
        print(f"  Exception msg  : {e}")
        print(f"  Full repr      : {repr(e)}")
        if hasattr(e, "status_code"):
            print(f"  HTTP status    : {e.status_code}")
        if hasattr(e, "response"):
            try:
                print(f"  Response body  : {e.response.text[:2000]}")
            except Exception:
                print(f"  Response obj   : {repr(e.response)}")
        if hasattr(e, "code"):
            print(f"  Error code     : {e.code}")
        if hasattr(e, "message"):
            print(f"  Error message  : {e.message}")
        sys.exit(1)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)
    elapsed = time.time() - t0
    print(f"Gemini responded in {elapsed:.1f}s", flush=True)
    # Dump raw response object attributes for debugging
    print(f"\n--- Response inspection ---")
    print(f"  type(response)       : {type(response).__name__}")
    print(f"  candidates count     : {len(response.candidates) if response.candidates else 0}")
    print(f"  prompt_feedback      : {response.prompt_feedback}")
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print(f"  usage_metadata       : {response.usage_metadata}")
    if response.candidates:
        candidate = response.candidates[0]
        print(f"  finish_reason        : {candidate.finish_reason}")
        if candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                print(f"    Safety: {rating.category} → {rating.probability}")
        # Show raw parts summary
        if candidate.content and candidate.content.parts:
            print(f"  parts count          : {len(candidate.content.parts)}")
            for pi, part in enumerate(candidate.content.parts):
                has_text = bool(getattr(part, "text", None))
                part_type = type(part).__name__
                text_len = len(part.text) if has_text else 0
                print(f"    part[{pi}]: type={part_type}, has_text={has_text}, text_len={text_len}")
        else:
            print("  WARNING: candidate has no content/parts!")
    else:
        print("  WARNING: No candidates returned by Gemini.")
    print("--- End response inspection ---\n")
    try:
        response_text = response.text
    except Exception as e:
        print(f"ERROR accessing response.text: {type(e).__name__}: {e}")
        print(f"  Full response repr (first 3000 chars): {repr(response)[:3000]}")
        sys.exit(1)
    if not response_text:
        print("ERROR: Gemini returned an empty response (no text).")
        print(f"  Full response repr (first 3000 chars): {repr(response)[:3000]}")
        sys.exit(1)
    # Parse masks
    masks = parse_segmentation_masks(response_text, img_height=height, img_width=width)
    print(f"Found {len(masks)} component(s).")
    # Assign numbered labels (LED1, diode2, resistor1, …)
    masks = assign_numbered_labels(masks)
    print("Assigned numbered labels:\n")
    # Print bounding-box info
    for i, m in enumerate(masks):
        print(f"  [{i}] {m.label}")
        print(f"       box (pixels): x0={m.x0}, y0={m.y0}, x1={m.x1}, y1={m.y1}")
        print(f"       size: {m.x1 - m.x0} x {m.y1 - m.y0} px\n")
    # Prepare crops output directory
    crops_dir = os.path.join(os.path.dirname(output_path) or ".", "crops")
    os.makedirs(crops_dir, exist_ok=True)
    # Save the original image into crops
    original_dest = os.path.join(crops_dir, "original.jpg")
    im_fullres.save(original_dest, "JPEG", quality=95)
    print(f"Saved original image → {original_dest}")
    # Draw the segmented overlay and save into crops
    result = draw_results(im, masks)
    segmented_dest = os.path.join(crops_dir, "segmented_output.jpg")
    result.convert("RGB").save(segmented_dest, "JPEG", quality=95)
    print(f"Saved segmented image → {segmented_dest}")
    # Run per-component analysis and save annotations.json
    print(f"Running component analysis …")
    analyze_components(im, masks, crops_dir, img_fullres=im_fullres, save_temps=save_temps)
    # Run breadboard pin detection
    breadboard_debug_path = os.path.join(crops_dir, "breadboard_debug.png")
    print(f"Running breadboard pin detection on {image_path} …")
    try:
        breadboard_detection.run_detector(
            image_path=image_path,
            debug_image_path=breadboard_debug_path,
            nearest_pool=40,
            reject_angle_low=5.0,
            reject_angle_high=85.0,
        )
        print(f"Saved breadboard debug image → {breadboard_debug_path}")
    except Exception as e:
        print(f"  [warn] breadboard detection failed: {type(e).__name__}: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment components in a breadboard image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("output_path", nargs="?", default="segmented_output.png", help="Path for the annotated output image")
    parser.add_argument("--save-temps", action="store_true", help="Save the temporary crop files passed to analysis scripts into the crops directory")
    args = parser.parse_args()
    main(args.image_path, args.output_path, save_temps=args.save_temps)