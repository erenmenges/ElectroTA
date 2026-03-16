#!/usr/bin/env python3
"""
Resistor Band Reading Direction Detector
Determines which end of a through-hole resistor to start reading the color
bands from.  Works by:
1.  Detecting the blue resistor body via HSV thresholding.
2.  Finding the principal axis with PCA on the blue pixels.
3.  Building a tight body mask from the convex hull of inlier blue pixels.
4.  Warping the image so the resistor axis is horizontal.
5.  Computing a 1-D "band intensity" profile along the axis.
6.  Locating bands as contiguous above-threshold regions.
7.  Identifying the tolerance band via the largest inter-band gap
    (with an edge-proximity tiebreaker when gaps are close).
8.  Returning the direction (top / bottom / left / right) to start reading
    from (opposite to the tolerance band).
"""
import cv2
import numpy as np
import os
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def preprocess(img):
    """Boost saturation, lift shadows, sharpen to make bands more vivid."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15 + 20, 0, 255)
    hsv = hsv.astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    sharp = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    return cv2.filter2D(out, -1, sharp)


def blue_mask(img):
    """Return a binary mask of blue pixels (the resistor body color)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([80, 25, 20]), np.array([140, 255, 255]))


def strict_blue_mask(img):
    """Return a mask of only confidently-blue pixels (high saturation + value)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([85, 80, 100]), np.array([130, 255, 255]))


def pca_axis(mask):
    """
    PCA on blue pixels → (center, major_axis, minor_axis, minor_std).
    major_axis is canonicalised to point roughly right / down.
    """
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    i_major = np.argmax(evals)
    i_minor = 1 - i_major
    major = evecs[:, i_major]
    minor = evecs[:, i_minor]
    # canonical direction
    if major[0] < -0.01 or (abs(major[0]) < 0.01 and major[1] < 0):
        major = -major
    return mean, major, minor, np.sqrt(evals[i_minor])


def body_mask_from_hull(mask, center, major, minor, minor_std):
    """
    Build a tight body mask:
      1. Project blue pixels onto the minor axis.
      2. Keep only inliers (within ±2·σ).
      3. Return the filled convex hull of those inliers.
    """
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centered = pts - center
    proj_minor = centered @ minor
    inlier = np.abs(proj_minor) < 2.0 * minor_std
    filtered = pts[inlier]
    if len(filtered) < 10:
        filtered = pts  # fall back
    hull = cv2.convexHull(filtered.astype(np.float32))
    body = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(body, [hull.astype(np.int32)], -1, 255, cv2.FILLED)
    return body


def warp_horizontal(img, mask, center, axis):
    """Rotate so that *axis* becomes horizontal → left-to-right."""
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    h, w = img.shape[:2]
    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return (cv2.warpAffine(img, M, (nw, nh)),
            cv2.warpAffine(mask, M, (nw, nh)))


def crop_to(img, mask, margin=5):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img, mask
    x1 = max(0, xs.min() - margin)
    x2 = min(img.shape[1], xs.max() + margin)
    y1 = max(0, ys.min() - margin)
    y2 = min(img.shape[0], ys.max() + margin)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def refine_body_with_strip(w_hull, w_strict):
    """
    When the hull covers nearly the entire image (blue pixels are everywhere),
    use the row-wise density of strict-blue pixels to find the actual body strip
    and intersect it with the hull mask.
    """
    nh, nw = w_hull.shape[:2]
    row_count = np.sum(w_strict > 0, axis=1).astype(np.float64)
    row_smooth = cv2.GaussianBlur(
        row_count.reshape(-1, 1).astype(np.float32), (1, 15), 0
    ).flatten()
    peak_row = int(np.argmax(row_smooth))
    peak_val = row_smooth[peak_row]
    # Body strip = rows where strict-blue density ≥ 30 % of peak
    above = row_smooth >= peak_val * 0.30
    r_start = peak_row
    while r_start > 0 and above[r_start - 1]:
        r_start -= 1
    r_end = peak_row
    while r_end < nh - 1 and above[r_end + 1]:
        r_end += 1
    # Add a 10 % margin
    margin = max(5, (r_end - r_start) // 10)
    r_start = max(0, r_start - margin)
    r_end = min(nh - 1, r_end + margin)
    strip = np.zeros_like(w_hull)
    strip[r_start:r_end + 1, :] = 255
    return cv2.bitwise_and(w_hull, strip)


# ── 1-D band profile ────────────────────────────────────────────────────────

def band_profile(crop_img, crop_mask):
    """Column-wise fraction of body pixels that are non-blue."""
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    is_blue = (h >= 80) & (h <= 140) & (s > 20)
    is_highlight = (s < 25) & (v > 170)
    is_body = crop_mask > 0
    is_band = is_body & ~is_blue & ~is_highlight
    width = crop_img.shape[1]
    profile = np.zeros(width, dtype=np.float64)
    body_n  = np.zeros(width, dtype=np.float64)
    for c in range(width):
        nb = np.sum(is_body[:, c])
        if nb:
            profile[c] = np.sum(is_band[:, c]) / nb
            body_n[c] = nb
    return profile, body_n


def find_band_regions(profile, body_n, threshold=0.12, min_span=2):
    """Find contiguous above-threshold regions in the smoothed profile."""
    max_b = body_n.max() or 1
    valid = body_n >= max_b * 0.25
    ksize = max(5, len(profile) // 30) | 1  # ensure odd
    smoothed = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32), (ksize, 1), 0
    ).flatten()
    smoothed[~valid] = 0
    above = smoothed > threshold
    regions, in_r, start = [], False, 0
    for i in range(len(above)):
        if above[i] and not in_r:
            start, in_r = i, True
        elif not above[i] and in_r:
            regions.append((start, i - 1)); in_r = False
    if in_r:
        regions.append((start, len(above) - 1))
    peaks = []
    for s, e in regions:
        if e - s + 1 < min_span:
            continue
        cols = np.arange(s, e + 1)
        w = smoothed[s:e + 1]
        peaks.append((float(np.average(cols, weights=w)), float(w.max())))
    return peaks, smoothed


# ── tolerance-band identification ────────────────────────────────────────────

def tolerance_side(peaks, total_len):
    """
    Return 'left' or 'right' (in warped horizontal space) for the tolerance band.
    Primary heuristic : side of the largest inter-band gap with fewer bands.
    Tiebreaker        : edge proximity (tolerance closer to body edge).
    """
    if len(peaks) < 2:
        if len(peaks) == 1:
            return 'left' if peaks[0][0] < total_len / 2 else 'right'
        return 'right'
    positions = [p[0] for p in peaks]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    edge_l = positions[0]
    edge_r = total_len - positions[-1]

    # Sort gap indices by gap size, descending
    sorted_gap_idx = sorted(range(len(gaps)), key=lambda k: gaps[k], reverse=True)
    best_idx = sorted_gap_idx[0]
    best_gap = gaps[best_idx]

    # Check if the second-largest gap is close (within 20% of the largest)
    ambiguous = False
    if len(sorted_gap_idx) > 1:
        second_gap = gaps[sorted_gap_idx[1]]
        if second_gap > best_gap * 0.80:
            ambiguous = True

    n_left  = best_idx + 1
    n_right = len(positions) - best_idx - 1

    if not ambiguous and n_left != n_right:
        # Clear winner from gap count
        return 'right' if n_right < n_left else 'left'

    # Ambiguous gap or equal band counts → use edge proximity.
    # The tolerance band is typically closer to the physical edge of the body.
    return 'left' if edge_l < edge_r else 'right'


# ── direction label ──────────────────────────────────────────────────────────

def reading_direction(axis, tol_side):
    """
    Map axis vector + tolerance side to a human direction (top / bottom /
    left / right) for where to START reading (opposite the tolerance band).
    """
    start = -axis if tol_side == 'right' else axis
    angle = np.degrees(np.arctan2(start[1], start[0]))
    if -45 <= angle <= 45:
        return "right"
    if 45 < angle <= 135:
        return "bottom"
    if -135 <= angle < -45:
        return "top"
    return "left"


# ── debug visualisation ─────────────────────────────────────────────────────

def _draw_debug_image(img, proc, bmask, center, major, minor, m_std,
                      body, w_img, w_mask, c_img, c_mask,
                      prof, smoothed, peaks, tol_side, direction,
                      hull_refined):
    """
    Build a single composite debug image with 6 panels:
      1. Original + PCA axis overlay
      2. Blue mask
      3. Body hull mask
      4. Warped + cropped body region
      5. Band detection overlay (non-blue pixels highlighted)
      6. 1-D profile plot with detected peaks and tolerance marker
    """
    # --- Resize helper (fit into a fixed cell) ---
    CELL_W, CELL_H = 400, 300

    def fit(image, label=None):
        """Resize image to fit CELL_W×CELL_H, pad with dark grey, add label."""
        if image is None or image.size == 0:
            canvas = np.full((CELL_H, CELL_W, 3), 40, dtype=np.uint8)
            cv2.putText(canvas, "N/A", (150, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return canvas
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        scale = min(CELL_W / w, CELL_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((CELL_H, CELL_W, 3), 40, dtype=np.uint8)
        y_off = (CELL_H - nh) // 2
        x_off = (CELL_W - nw) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
        if label:
            cv2.putText(canvas, label, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
                        cv2.LINE_AA)
        return canvas

    # Panel 1: Original + PCA axis + center
    p1 = img.copy()
    cx, cy = int(center[0]), int(center[1])
    length = int(max(img.shape[:2]) * 0.35)
    pt1 = (cx - int(major[0] * length), cy - int(major[1] * length))
    pt2 = (cx + int(major[0] * length), cy + int(major[1] * length))
    cv2.line(p1, pt1, pt2, (0, 255, 0), 2)  # major axis green
    pt3 = (cx - int(minor[0] * length // 2), cy - int(minor[1] * length // 2))
    pt4 = (cx + int(minor[0] * length // 2), cy + int(minor[1] * length // 2))
    cv2.line(p1, pt3, pt4, (0, 0, 255), 1)  # minor axis red
    cv2.circle(p1, (cx, cy), 5, (255, 0, 255), -1)

    # Panel 2: Blue mask
    p2 = bmask.copy()

    # Panel 3: Body hull mask (with refinement note)
    hull_label = "3. Body mask" + (" (refined)" if hull_refined else " (hull)")
    p3 = body.copy()

    # Panel 4: Warped + cropped with mask outline
    p4 = c_img.copy()
    contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p4, contours, -1, (0, 255, 0), 1)

    # Panel 5: Band pixels highlighted in red within cropped body
    p5 = c_img.copy()
    hsv_c = cv2.cvtColor(c_img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv_c[:, :, 0], hsv_c[:, :, 1], hsv_c[:, :, 2]
    is_blue = (h_ch >= 80) & (h_ch <= 140) & (s_ch > 20)
    is_highlight = (s_ch < 25) & (v_ch > 170)
    is_body = c_mask > 0
    is_band = is_body & ~is_blue & ~is_highlight
    p5[is_band] = [0, 0, 255]  # mark band pixels red

    # Draw detected peak positions as vertical lines
    for pk_pos, pk_val in peaks:
        x = int(pk_pos)
        if 0 <= x < p5.shape[1]:
            cv2.line(p5, (x, 0), (x, p5.shape[0] - 1), (0, 255, 0), 1)

    # Panel 6: 1-D profile plot
    plot_h, plot_w = CELL_H, CELL_W
    p6 = np.full((plot_h, plot_w, 3), 40, dtype=np.uint8)
    if len(smoothed) > 0:
        # Draw axes
        margin_l, margin_b, margin_t = 40, 30, 30
        gw = plot_w - margin_l - 10
        gh = plot_h - margin_b - margin_t
        cv2.line(p6, (margin_l, margin_t), (margin_l, margin_t + gh),
                 (150, 150, 150), 1)
        cv2.line(p6, (margin_l, margin_t + gh),
                 (margin_l + gw, margin_t + gh), (150, 150, 150), 1)

        max_val = max(smoothed.max(), 0.01)
        n = len(smoothed)
        # Draw smoothed profile
        pts = []
        for i in range(n):
            x = margin_l + int(i / max(n - 1, 1) * gw)
            y = margin_t + gh - int(smoothed[i] / max_val * gh)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(p6, pts[i], pts[i + 1], (255, 200, 0), 1, cv2.LINE_AA)

        # Draw threshold line
        thresh_y = margin_t + gh - int(0.12 / max_val * gh)
        cv2.line(p6, (margin_l, thresh_y), (margin_l + gw, thresh_y),
                 (0, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(p6, "thr", (margin_l + gw - 30, thresh_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)

        # Draw peak markers
        for pk_pos, pk_val in peaks:
            x = margin_l + int(pk_pos / max(n - 1, 1) * gw)
            y = margin_t + gh - int(pk_val / max_val * gh)
            cv2.circle(p6, (x, y), 4, (0, 255, 0), -1)

        # Mark tolerance side
        tol_x = margin_l + 5 if tol_side == 'left' else margin_l + gw - 15
        cv2.putText(p6, "TOL", (tol_x, margin_t + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # Mark reading start side
        read_x = margin_l + gw - 40 if tol_side == 'left' else margin_l + 5
        cv2.putText(p6, "READ>>", (read_x, margin_t + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Assemble 3×2 grid
    row1 = np.hstack([fit(p1, "1. Original + PCA"),
                      fit(p2, "2. Blue mask"),
                      fit(p3, hull_label)])
    row2 = np.hstack([fit(p4, "4. Warped crop + mask"),
                      fit(p5, "5. Band pixels (red)"),
                      fit(p6, f"6. Profile → {direction}")])

    # Add a summary bar at the bottom
    bar = np.full((36, CELL_W * 3, 3), 30, dtype=np.uint8)
    summary = (f"Bands detected: {len(peaks)} | "
               f"Tolerance side: {tol_side} | "
               f"Reading direction: {direction}")
    cv2.putText(bar, summary, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                cv2.LINE_AA)

    return np.vstack([row1, row2, bar])


# ── main pipeline ────────────────────────────────────────────────────────────

def detect_reading_direction(image_path, debug=True):
    """
    Given an image of a resistor, return the direction to start reading
    the color bands from.  One of: 'top', 'bottom', 'left', 'right'.

    Parameters
    ----------
    image_path : str or Path
        Path to the resistor image.
    debug : bool
        When True, saves:
          - The original image to  raw_resistor/<filename>
          - An annotated debug image to  raw_resistor/<filename>_debug.png
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return "error: could not load image"

    # ── Debug: save raw image ────────────────────────────────────────────
    if debug:
        debug_dir = Path("raw_resistor")
        debug_dir.mkdir(exist_ok=True)
        stem = Path(image_path).stem
        ext = Path(image_path).suffix or ".png"
        cv2.imwrite(str(debug_dir / f"{stem}{ext}"), img)

    proc = preprocess(img)
    bmask = blue_mask(proc)
    if np.sum(bmask > 0) < 100:
        return "error: not enough blue pixels"

    center, major, minor, m_std = pca_axis(bmask)
    body = body_mask_from_hull(bmask, center, major, minor, m_std)

    # Warp the image and body mask so the resistor axis is horizontal
    w_img, w_mask = warp_horizontal(proc, body, center, major)

    # If the hull covers almost the entire crop it is too loose (blue
    # pixels from the background inflate it).  Refine with a horizontal
    # strip derived from strict-blue row density.
    c_img, c_mask = crop_to(w_img, w_mask)
    hull_coverage = np.sum(c_mask > 0) / max(c_mask.size, 1)
    hull_refined = False
    if hull_coverage > 0.95:
        sbmask = strict_blue_mask(proc)
        _, w_strict = warp_horizontal(proc, sbmask, center, major)
        w_mask = refine_body_with_strip(w_mask, w_strict)
        c_img, c_mask = crop_to(w_img, w_mask)
        hull_refined = True

    prof, bn = band_profile(c_img, c_mask)
    peaks, smoothed = find_band_regions(prof, bn)
    ts = tolerance_side(peaks, len(prof))
    direction = reading_direction(major, ts)

    # ── Debug: save annotated composite ──────────────────────────────────
    if debug:
        debug_img = _draw_debug_image(
            img, proc, bmask, center, major, minor, m_std,
            body, w_img, w_mask, c_img, c_mask,
            prof, smoothed, peaks, ts, direction,
            hull_refined,
        )
        cv2.imwrite(str(debug_dir / f"{stem}_debug.png"), debug_img)
        print(f"[debug] saved raw + debug images to {debug_dir}/")

    print(f"start from: {direction}")
    return direction


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    enable_debug = "--debug" in args
    if enable_debug:
        args.remove("--debug")
    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} [--debug] <image_path>", file=sys.stderr)
        sys.exit(1)
    print(detect_reading_direction(args[0], debug=enable_debug))