import cv2
import numpy as np
import sys
import os


def detect_cathode_direction(image_path):
    """
    Detects the cathode direction of a diode by locating the white stripe.
    Returns: 'left', 'right', 'top', or 'bottom'
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Step 1: Find the diode body (dark region) using Otsu threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, body_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 2: Find the largest contour (the diode body)
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No diode body found")

    diode_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(diode_contour)

    # Step 3: Determine orientation based on aspect ratio
    is_horizontal = bw > bh

    # Step 4: Create a mask of just the diode body region
    diode_mask = np.zeros_like(gray)
    cv2.drawContours(diode_mask, [diode_contour], -1, 255, -1)

    # Step 5: Find the white stripe within the diode body
    # The white stripe is the brightest region inside the diode
    # Use a high threshold to isolate just the bright stripe
    diode_region = cv2.bitwise_and(gray, gray, mask=diode_mask)

    # Compute a brightness threshold for the stripe
    # Look at pixels inside the diode body only
    diode_pixels = gray[diode_mask > 0]
    if len(diode_pixels) == 0:
        raise ValueError("No pixels in diode mask")

    # The white stripe should be significantly brighter than the dark body
    mean_brightness = np.mean(diode_pixels)
    std_brightness = np.std(diode_pixels)
    stripe_threshold = mean_brightness + std_brightness * 0.8

    # Also ensure minimum brightness
    stripe_threshold = max(stripe_threshold, 140)

    _, stripe_mask = cv2.threshold(diode_region, stripe_threshold, 255, cv2.THRESH_BINARY)

    # Clean up stripe mask
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
    stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_CLOSE, small_kernel, iterations=1)

    # Step 6: Find the centroid of the stripe pixels
    stripe_points = np.where(stripe_mask > 0)
    if len(stripe_points[0]) == 0:
        # Fallback: lower threshold
        stripe_threshold = mean_brightness + std_brightness * 0.3
        _, stripe_mask = cv2.threshold(diode_region, stripe_threshold, 255, cv2.THRESH_BINARY)
        stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
        stripe_points = np.where(stripe_mask > 0)

    if len(stripe_points[0]) == 0:
        raise ValueError("Could not find white stripe")

    stripe_cy = np.mean(stripe_points[0])  # row = y
    stripe_cx = np.mean(stripe_points[1])  # col = x

    # Step 7: Compare stripe centroid to diode body centroid
    M = cv2.moments(diode_contour)
    if M["m00"] == 0:
        diode_cx = x + bw / 2
        diode_cy = y + bh / 2
    else:
        diode_cx = M["m10"] / M["m00"]
        diode_cy = M["m01"] / M["m00"]

    dx = stripe_cx - diode_cx
    dy = stripe_cy - diode_cy

    # Step 8: Determine direction based on orientation and offset
    if is_horizontal:
        return "right" if dx > 0 else "left"
    else:
        return "bottom" if dy > 0 else "top"


if __name__ == "__main__":
    image_dir = "/mnt/user-data/uploads"
    test_cases = [
        ("diode1_temp.jpg", "right"),
        ("diode2_temp.jpg", "left"),
        ("diode3_temp.jpg", "top"),
        ("diode4_temp.jpg", "bottom"),
    ]

    print("=" * 50)
    print("Diode Cathode Direction Detection")
    print("=" * 50)

    all_passed = True
    for i, (filename, expected) in enumerate(test_cases, 1):
        path = os.path.join(image_dir, filename)
        try:
            result = detect_cathode_direction(path)
            status = "PASS" if result == expected else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"Diode {i}: detected='{result}', expected='{expected}' [{status}]")
        except Exception as e:
            all_passed = False
            print(f"Diode {i}: ERROR - {e} [FAIL]")

    print("=" * 50)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")