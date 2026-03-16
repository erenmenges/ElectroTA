import sys
import cv2
import numpy as np

def detect_cathode_direction(image_path):
    """
    Detects the direction of the cathode (white stripe) on a capacitor.
    Uses a Contour Vector approach: isolates the boundary of the bright region
    and leverages symmetry. The symmetrical circular top cancels its own vectors out,
    leaving only the asymmetrical stripe to pull the vector exactly in its direction.
    Returns 8-way directions: 'top-left', 'top', 'top-right', 'left', 'right', 
    'bottom-left', 'bottom', or 'bottom-right'.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image '{image_path}' could not be loaded.")

    # Convert to grayscale and apply slight blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Isolate bright regions
    _, bright_mask = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    
    # 2. Find the single largest contiguous bright blob (Silver Top + White Stripe)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("unknown")
        return "unknown"
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 100:
        print("unknown")
        return "unknown"
        
    # 3. Find the centroid of the contour using image moments
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        print("unknown")
        return "unknown"
        
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # 4. Calculate a contour-based weighted vector
    # Reshape contour points to an easily iterable array of (x, y) coordinates
    contour_pts = largest_contour.reshape(-1, 2) 
    
    x_coords = contour_pts[:, 0]
    y_coords = contour_pts[:, 1]
    
    dx = x_coords - cx
    dy = y_coords - cy
    
    # Calculate distance of each contour point from the centroid
    distances = np.sqrt(dx**2 + dy**2)
    
    # Weight each vector by the square of its distance.
    # This massively amplifies points that protrude outwards (the stripe)
    # while the symmetrical parts (the circle edges) naturally cancel each other out.
    weights = distances ** 2
    
    vec_x = np.sum(dx * weights)
    vec_y = np.sum(dy * weights)
    
    if vec_x == 0 and vec_y == 0:
        print("unknown")
        return "unknown"
        
    # Calculate final angle in degrees
    angle_deg = np.degrees(np.arctan2(vec_y, vec_x))
    
    # Map angle to 8-way directions
    if -22.5 <= angle_deg < 22.5: 
        direction = "right"
    elif 22.5 <= angle_deg < 67.5: 
        direction = "bottom-right"
    elif 67.5 <= angle_deg < 112.5: 
        direction = "bottom"
    elif 112.5 <= angle_deg < 157.5: 
        direction = "bottom-left"
    elif angle_deg >= 157.5 or angle_deg < -157.5: 
        direction = "left"
    elif -157.5 <= angle_deg < -112.5: 
        direction = "top-left"
    elif -112.5 <= angle_deg < -67.5: 
        direction = "top"
    elif -67.5 <= angle_deg < -22.5: 
        direction = "top-right"
    else:
        direction = "unknown"

    print(direction)
    return direction

if __name__ == "__main__":
    # Read the image path from the command line argument if provided
    image_file = sys.argv[1] if len(sys.argv) > 1 else 'capacitor_0.jpg'
    
    try:
        direction = detect_cathode_direction(image_file)
        # Note: The function itself now prints the result so it will be visible 
        # when imported and called from other scripts. 
        # Left this print here untouched to strictly preserve everything else.
    except Exception as e:
        print(f"Error processing image: {e}")