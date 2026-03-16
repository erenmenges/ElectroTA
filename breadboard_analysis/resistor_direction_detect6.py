import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os
import sys

def enhance_image(img):
    """
    Pre-processes the image by increasing saturation, fixing shadows/highlights,
    and sharpening the bands as requested.
    """
    # 1. Increase Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    hsv = hsv.astype(np.uint8)
    img_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Adjust shadows/highlights (CLAHE on L channel of LAB color space)
    lab = cv2.cvtColor(img_sat, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. Increase Sharpness (Unsharp mask)
    blur = cv2.GaussianBlur(img_contrast, (5, 5), 0)
    img_sharp = cv2.addWeighted(img_contrast, 1.5, blur, -0.5, 0)

    return img_sharp

def determine_reading_direction(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return f"Could not read image: {image_path}"

    enhanced = enhance_image(img)

    # Isolate the blue body of the resistor
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 20, 20])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find the contour of the resistor
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No resistor found"

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Determine the major axis
    d01 = np.linalg.norm(box[0] - box[1])
    d12 = np.linalg.norm(box[1] - box[2])

    if d01 > d12:
        m1 = (box[1] + box[2]) / 2
        m2 = (box[0] + box[3]) / 2
        minor_len = d12
    else:
        m1 = (box[0] + box[1]) / 2
        m2 = (box[2] + box[3]) / 2
        minor_len = d01

    A = m1
    B = m2
    
    # Sort endpoints to ensure A is always consistently the "left" or "top" point
    if A[0] > B[0] or (abs(A[0] - B[0]) < 1e-5 and A[1] > B[1]):
        A, B = B, A

    # --- Setup Debug Image ---
    debug_img = enhanced.copy()
    cv2.drawContours(debug_img, [box], 0, (0, 0, 255), 2) # Bounding box in red

    # Because the blue mask often gets broken by the edge bands, the bounding box
    # can be too short. We extend the A->B segment outwards to guarantee we capture all bands.
    length = np.linalg.norm(B - A)
    v_major = (B - A) / length
    v_minor = np.array([-v_major[1], v_major[0]])

    extend_factor = 0.5 # Extend by 50% of the box's length on both sides
    A_ext = A - v_major * (length * extend_factor)
    B_ext = B + v_major * (length * extend_factor)

    # Extract the 1D profile of intensities along the extended major axis
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    num_points = 800 # Higher resolution for the longer extended line
    swath = max(1, int(minor_len * 0.2)) # Sample a central swath to reduce glare/noise

    profile = []
    valid_pts = [] # Keep track of points that actually fall inside the image boundaries
    h, w = gray.shape
    for i in range(num_points):
        t = i / (num_points - 1)
        pt = A_ext * (1 - t) + B_ext * t
        
        val, count = 0, 0
        for w_idx in range(-swath, swath + 1):
            sample_pt = pt + w_idx * v_minor
            x, y = int(sample_pt[0]), int(sample_pt[1])
            if 0 <= y < h and 0 <= x < w:
                val += float(gray[y, x]) # Cast to float
                count += 1
                
        if count > 0:
            profile.append(val / count)
            valid_pts.append(pt)

    if not valid_pts:
        return "Failed to sample resistor body."

    profile = np.array(profile)
    
    # Re-assign A and B to the true boundaries of our sampled profile 
    # to ensure directions map correctly back to the image.
    A_sampled = valid_pts[0]
    B_sampled = valid_pts[-1]

    # Draw the new extended sampling line
    cv2.line(debug_img, (int(A_sampled[0]), int(A_sampled[1])), (int(B_sampled[0]), int(B_sampled[1])), (0, 255, 255), 1)
    cv2.circle(debug_img, (int(A_sampled[0]), int(A_sampled[1])), 5, (0, 255, 0), -1)
    cv2.circle(debug_img, (int(B_sampled[0]), int(B_sampled[1])), 5, (255, 0, 0), -1)

    # Find the bands (which appear as dark valleys/minima in the profile)
    inv_profile = 255 - profile
    inv_profile = gaussian_filter1d(inv_profile, sigma=3) # Smooth out noise
    
    prom = np.max(inv_profile) * 0.05
    peaks, properties = find_peaks(inv_profile, prominence=prom, distance=25)

    # Use the fact that these are exactly 5-band resistors:
    # If the extended line picked up noise from the background, we filter it out 
    # by taking strictly the 5 most prominent valleys (bands).
    if len(peaks) > 5:
        prominences = properties['prominences']
        top_5_idx = np.argsort(prominences)[-5:]
        peaks = np.sort(peaks[top_5_idx]) # Re-sort back into spatial order

    # Plot detected peaks as pink dots on the debug image
    for p in peaks:
        pt = valid_pts[p]
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1)

    # Save the debug image out to disk
    base_name = os.path.basename(image_path)
    debug_filename = f"debug_{base_name}"
    cv2.imwrite(debug_filename, debug_img)

    if len(peaks) < 3:
        return f"Could not detect enough bands to determine direction. Found {len(peaks)}. Check {debug_filename}!"

    # Analyze gaps between bands
    gaps = np.diff(peaks)
    max_gap_idx = np.argmax(gaps)

    # If the largest gap is in the first half of the bands, 
    # the tolerance band is on side A, so we start reading from side B.
    if max_gap_idx < len(gaps) / 2:
        start_pt = B_sampled
    else:
        start_pt = A_sampled

    # Convert the starting point into a relative direction text
    center = (A_sampled + B_sampled) / 2
    dir_vec = start_pt - center
    dx, dy = dir_vec
    
    abs_dx = abs(dx)
    abs_dy = abs(dy) + 1e-6 # prevent div by zero

    directions = []
    if abs_dx > 2 * abs_dy: # mostly horizontal
        directions.append("right" if dx > 0 else "left")
    elif abs_dy > 2 * abs_dx: # mostly vertical
        directions.append("bottom" if dy > 0 else "top")
    else: # diagonal
        directions.append("right" if dx > 0 else "left")
        directions.append("bottom" if dy > 0 else "top")

    if len(directions) == 1:
        return directions[0]
    else:
        print(f"directions: {directions[0]} - {directions[1]}")
        return f"{directions[0]} - {directions[1]}"

def main(image_path):
    """
    Entry point for external callers (e.g. get_resistor_analysis.py).
    Returns a direction string like "left", "right", "left - bottom", etc.
    Returns "no resistor detected" or "no bands detected" on failure.
    """
    if not os.path.exists(image_path):
        return "no resistor detected"

    result = determine_reading_direction(image_path)


    if result is None:
        return "no resistor detected"

    result_lower = result.lower()

    if "no resistor" in result_lower or "could not read" in result_lower:
        return "no resistor detected"

    if "not detect enough bands" in result_lower or "failed to sample" in result_lower:
        return "no bands detected"

    return result


if __name__ == "__main__":
    # If file paths are provided via command line arguments, use them
    if len(sys.argv) > 1:
        images = sys.argv[1:]
    else:
        # Fallback to defaults if run without arguments
        images = [
            "resistor1_temp.jpg",
            "resistor2_temp.jpg",
            "resistor3_temp.jpg",
            "resistor4_temp.jpg",
            "resistor5_temp.jpg"
        ]
    
    for img_name in images:
        if os.path.exists(img_name):
            direction = determine_reading_direction(img_name)
            print(f"{img_name} should say: \"{direction}\"")
        else:
            print(f"File {img_name} not found.")