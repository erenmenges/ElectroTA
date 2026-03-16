#!/usr/bin/env python3
"""
breadboard_pin_detector.py

Visible breadboard pin detector with cyan-dot visualization and
cyan neighbor-line visualization.

This version keeps the cyan dot detector unchanged and adds:
- for each cyan dot, inspect the 15 nearest neighbors
- draw up to 10 non-diagonal cyan lines
- reject neighbors whose angle is too diagonal
- rejection angle range is tunable from CLI
- callable from other python scripts via run_detector()

Requirements:
    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


def odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def cluster_1d(values: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge nearby 1D coordinates into clusters.
    Returns:
        centers, counts
    """
    if len(values) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    v = np.sort(values.astype(np.float32))
    groups = []
    counts = []

    cur = [float(v[0])]
    for x in v[1:]:
        if x - cur[-1] <= tol:
            cur.append(float(x))
        else:
            groups.append(float(np.median(cur)))
            counts.append(len(cur))
            cur = [float(x)]

    groups.append(float(np.median(cur)))
    counts.append(len(cur))

    return np.array(groups, dtype=np.float32), np.array(counts, dtype=np.int32)


def rotate_points(points: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    R = np.array(
        [
            [math.cos(a), -math.sin(a)],
            [math.sin(a),  math.cos(a)],
        ],
        dtype=np.float32,
    )
    return (points - center) @ R.T + center


@dataclass
class DetectionDebug:
    board_mask: np.ndarray
    normalized_l: np.ndarray
    blackhat: np.ndarray
    hole_binary: np.ndarray
    hole_points: np.ndarray
    rotation_deg: float
    rotated_points: np.ndarray
    row_centers_rot: np.ndarray
    col_centers_rot: np.ndarray


class BreadboardPinDetector:
    def __init__(self) -> None:
        pass

    def _segment_board(self, image: np.ndarray) -> np.ndarray:
        """
        Broad segmentation of the breadboard body using:
        - high luminance
        - low chroma
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        chroma = np.sqrt(
            (A.astype(np.float32) - 128.0) ** 2 +
            (B.astype(np.float32) - 128.0) ** 2
        )

        l_thr = max(140, int(np.percentile(L, 55)))
        board = ((L >= l_thr) & (chroma <= 35.0)).astype(np.uint8) * 255

        k = odd(min(image.shape[:2]) * 0.03)
        k2 = max(3, odd(k * 0.5))

        board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        board = cv2.morphologyEx(board, cv2.MORPH_OPEN, np.ones((k2, k2), np.uint8))
        board = keep_largest_component(board)

        board = cv2.dilate(board, np.ones((5, 5), np.uint8), iterations=1)
        return board

    def _normalize_luminance(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce shadow / illumination variation while preserving local dark holes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)

        sigma = max(9.0, min(image.shape[:2]) / 18.0)
        bg = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)

        norm = (g.astype(np.float32) + 1.0) / (bg.astype(np.float32) + 1.0)
        norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return norm

    def _detect_hole_candidates(
        self,
        image: np.ndarray,
        board_mask: np.ndarray,
        normalized_l: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detect visible breadboard holes using black-hat morphology.

        Returns:
            blackhat image,
            thresholded binary hole map,
            bounding boxes of accepted hole candidates
        """
        h, w = image.shape[:2]
        k = odd(max(5, min(h, w) * 0.012))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        blackhat = cv2.morphologyEx(normalized_l, cv2.MORPH_BLACKHAT, kernel)
        blackhat = cv2.bitwise_and(blackhat, blackhat, mask=board_mask)

        vals = blackhat[board_mask > 0]
        if len(vals) == 0:
            return blackhat, np.zeros_like(blackhat), []

        thr_val, _ = cv2.threshold(
            vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thr_val = max(8, int(thr_val))

        hole_bin = (blackhat >= thr_val).astype(np.uint8) * 255
        hole_bin = cv2.morphologyEx(hole_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        hole_bin = cv2.bitwise_and(hole_bin, hole_bin, mask=board_mask)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(hole_bin, connectivity=8)

        boxes: List[Tuple[int, int, int, int]] = []
        for i in range(1, num):
            x, y, ww, hh, area = stats[i]

            if area < 4:
                continue
            if ww < 2 or hh < 2:
                continue
            if area > k * k * 4:
                continue

            aspect = ww / max(hh, 1e-6)
            fill = area / max(float(ww * hh), 1.0)

            if aspect < 0.35 or aspect > 3.0:
                continue
            if fill < 0.20:
                continue

            boxes.append((x, y, ww, hh))

        return blackhat, hole_bin, boxes

    def _boxes_to_points(self, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        pts = []
        for x, y, w, h in boxes:
            pts.append([x + 0.5 * w, y + 0.5 * h])
        return np.array(pts, dtype=np.float32)

    def _estimate_rotation(self, pts: np.ndarray) -> float:
        """
        Estimate breadboard long-axis angle from detected hole centers.
        Used only for relative row/col indexing of cyan dots.
        """
        if len(pts) < 4:
            return 0.0

        center = np.mean(pts, axis=0)
        P = pts - center
        cov = np.cov(P.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        angle = math.degrees(math.atan2(float(v[1]), float(v[0])))

        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        return angle

    def _assign_relative_indices(
        self,
        pts: np.ndarray,
        hole_scale: float,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign relative row/col indices to detected cyan dots only.
        No synthetic points are created.
        """
        if len(pts) == 0:
            return (
                0.0,
                np.empty((0, 2), dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        cluster_tol = max(2.0, 0.9 * hole_scale)
        rotation_deg = self._estimate_rotation(pts)
        center = np.mean(pts, axis=0)
        pts_rot = rotate_points(pts, center, -rotation_deg)

        row_centers, row_counts = cluster_1d(pts_rot[:, 1], tol=cluster_tol)
        col_centers, col_counts = cluster_1d(pts_rot[:, 0], tol=cluster_tol)

        if len(row_counts) > 0:
            row_keep = row_counts >= max(1, int(np.percentile(row_counts, 15)))
            row_centers = row_centers[row_keep]

        if len(col_counts) > 0:
            col_keep = col_counts >= max(1, int(np.percentile(col_counts, 15)))
            col_centers = col_centers[col_keep]

        row_centers = np.sort(row_centers.astype(np.float32))
        col_centers = np.sort(col_centers.astype(np.float32))

        return rotation_deg, pts_rot, row_centers, col_centers

    def detect(self, image: np.ndarray) -> Tuple[Dict, DetectionDebug]:
        board_mask = self._segment_board(image)
        normalized_l = self._normalize_luminance(image)
        blackhat, hole_bin, boxes = self._detect_hole_candidates(image, board_mask, normalized_l)

        if len(boxes) == 0:
            raise RuntimeError("No visible breadboard holes detected.")

        pts = self._boxes_to_points(boxes)

        hole_sizes = np.array([max(b[2], b[3]) for b in boxes], dtype=np.float32)
        hole_scale = float(np.median(hole_sizes)) if len(hole_sizes) else 6.0

        rotation_deg, pts_rot, row_centers, col_centers = self._assign_relative_indices(pts, hole_scale)

        visible_pins = []
        if len(row_centers) > 0 and len(col_centers) > 0:
            for i, (p, pr) in enumerate(zip(pts, pts_rot)):
                row = int(np.argmin(np.abs(row_centers - pr[1])))
                col = int(np.argmin(np.abs(col_centers - pr[0])))

                visible_pins.append(
                    {
                        "id": int(i),
                        "x": float(round(float(p[0]), 3)),
                        "y": float(round(float(p[1]), 3)),
                        "row": row,
                        "col": col,
                    }
                )
        else:
            for i, p in enumerate(pts):
                visible_pins.append(
                    {
                        "id": int(i),
                        "x": float(round(float(p[0]), 3)),
                        "y": float(round(float(p[1]), 3)),
                    }
                )

        result = {
            "pin_count": int(len(visible_pins)),
            "rotation_deg": float(round(rotation_deg, 4)),
            "visible_pins": visible_pins,
        }

        debug = DetectionDebug(
            board_mask=board_mask,
            normalized_l=normalized_l,
            blackhat=blackhat,
            hole_binary=hole_bin,
            hole_points=pts,
            rotation_deg=rotation_deg,
            rotated_points=pts_rot,
            row_centers_rot=row_centers,
            col_centers_rot=col_centers,
        )

        return result, debug

    def build_neighbor_lines(
        self,
        result: Dict,
        nearest_pool: int = 15,
        max_lines_per_point: int = 10,
        reject_angle_low: float = 30.0,
        reject_angle_high: float = 60.0,
    ) -> List[Dict]:
        """
        For each cyan point:
        - inspect the `nearest_pool` nearest other points
        - reject diagonal-ish neighbors
        - keep up to `max_lines_per_point` accepted neighbors

        Angle logic:
        - compute angle to horizontal using atan2(|dy|, |dx|)
        - angle is in [0, 90]
        - reject if reject_angle_low <= angle <= reject_angle_high
        """
        pins = result["visible_pins"]
        if len(pins) < 2:
            return []

        pts = np.array([[p["x"], p["y"]] for p in pins], dtype=np.float32)
        n = len(pts)

        lines: List[Dict] = []

        for i in range(n):
            diff = pts - pts[i]
            dist2 = np.sum(diff * diff, axis=1)
            order = np.argsort(dist2)

            accepted = 0
            examined = 0

            for j in order:
                if j == i:
                    continue

                examined += 1
                if examined > nearest_pool:
                    break

                dx = float(pts[j, 0] - pts[i, 0])
                dy = float(pts[j, 1] - pts[i, 1])

                angle = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-12))

                # Reject diagonal-ish angles only
                if reject_angle_low <= angle <= reject_angle_high:
                    continue

                lines.append(
                    {
                        "src_id": int(pins[i]["id"]),
                        "dst_id": int(pins[j]["id"]),
                        "x1": float(round(float(pts[i, 0]), 3)),
                        "y1": float(round(float(pts[i, 1]), 3)),
                        "x2": float(round(float(pts[j, 0]), 3)),
                        "y2": float(round(float(pts[j, 1]), 3)),
                        "angle_deg_from_horizontal": float(round(angle, 3)),
                    }
                )

                accepted += 1
                if accepted >= max_lines_per_point:
                    break

        return lines

    def _find_line_intersections(
        self, lines: List[Dict]
    ) -> List[Tuple[float, float]]:
        """Find points where line segments geometrically cross each other."""
        if not lines:
            return []

        # Deduplicate bidirectional lines (A->B and B->A kept once)
        seen_pairs: set = set()
        unique_lines: List[Dict] = []
        for line in lines:
            pair = (min(line["src_id"], line["dst_id"]), max(line["src_id"], line["dst_id"]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_lines.append(line)

        # Split into horizontal-ish and vertical-ish
        h_lines: List[Dict] = []
        v_lines: List[Dict] = []
        for line in unique_lines:
            if line["angle_deg_from_horizontal"] < 45:
                h_lines.append(line)
            else:
                v_lines.append(line)

        raw_intersections: List[Tuple[float, float]] = []
        eps = 1.5

        for h in h_lines:
            hx1, hy1 = h["x1"], h["y1"]
            hx2, hy2 = h["x2"], h["y2"]

            for v in v_lines:
                vx1, vy1 = v["x1"], v["y1"]
                vx2, vy2 = v["x2"], v["y2"]

                # Skip if they share an endpoint
                if abs(hx1 - vx1) < eps and abs(hy1 - vy1) < eps:
                    continue
                if abs(hx1 - vx2) < eps and abs(hy1 - vy2) < eps:
                    continue
                if abs(hx2 - vx1) < eps and abs(hy2 - vy1) < eps:
                    continue
                if abs(hx2 - vx2) < eps and abs(hy2 - vy2) < eps:
                    continue

                denom = (hx1 - hx2) * (vy1 - vy2) - (hy1 - hy2) * (vx1 - vx2)
                if abs(denom) < 1e-10:
                    continue

                t = ((hx1 - vx1) * (vy1 - vy2) - (hy1 - vy1) * (vx1 - vx2)) / denom
                u = -((hx1 - hx2) * (hy1 - vy1) - (hy1 - hy2) * (hx1 - vx1)) / denom

                if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                    ix = hx1 + t * (hx2 - hx1)
                    iy = hy1 + t * (hy2 - hy1)
                    raw_intersections.append((ix, iy))

        # Deduplicate nearby intersection points
        kept: List[Tuple[float, float]] = []
        dedup_eps = 3.0
        for ix, iy in raw_intersections:
            is_dup = False
            for kx, ky in kept:
                if abs(ix - kx) < dedup_eps and abs(iy - ky) < dedup_eps:
                    is_dup = True
                    break
            if not is_dup:
                kept.append((ix, iy))

        return kept

    def draw_debug(
        self,
        image: np.ndarray,
        result: Dict,
        lines: List[Dict] | None = None,
        draw_labels: bool = False,
        dot_radius: int = 2,
        line_thickness: int = 1,
    ) -> np.ndarray:
        """
        Draw only:
        - cyan dots
        - cyan neighbor lines
        """
        out = image.copy()

        if lines is not None:
            for line in lines:
                x1 = int(round(line["x1"]))
                y1 = int(round(line["y1"]))
                x2 = int(round(line["x2"]))
                y2 = int(round(line["y2"]))
                cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), line_thickness, cv2.LINE_AA)

        for p in result["visible_pins"]:
            x = int(round(p["x"]))
            y = int(round(p["y"]))

            cv2.circle(out, (x, y), dot_radius, (255, 255, 0), -1)

            if draw_labels and "row" in p and "col" in p:
                cv2.putText(
                    out,
                    f'{p["row"]},{p["col"]}',
                    (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        if lines is not None:
            intersections = self._find_line_intersections(lines)
            for ix, iy in intersections:
                cx = int(round(ix))
                cy = int(round(iy))
                cv2.circle(out, (cx, cy), dot_radius + 2, (0, 0, 255), -1)

        return out


def save_debug_panels(path: str, dbg_img: np.ndarray, debug: DetectionDebug, save_other: bool = False) -> None:
    cv2.imwrite(path, dbg_img)

    if save_other:
        base = path.rsplit(".", 1)[0]
        cv2.imwrite(base + "_board_mask.png", debug.board_mask)
        cv2.imwrite(base + "_normalized_l.png", debug.normalized_l)
        cv2.imwrite(base + "_blackhat.png", debug.blackhat)
        cv2.imwrite(base + "_hole_binary.png", debug.hole_binary)


def run_detector(
    image_path: str,
    json_path: str = "",
    debug_image_path: str = "",
    draw_labels: bool = False,
    nearest_pool: int = 15,
    max_lines_per_point: int = 10,
    reject_angle_low: float = 30.0,
    reject_angle_high: float = 60.0,
    dot_radius: int = 2,
    line_thickness: int = 1,
    do_print: bool = False,
    save_other_files: bool = False
) -> str:
    """
    Main entry point for processing an image programmatically.
    Returns the path to the main debug image (if requested).
    """
    if nearest_pool < 1:
        raise ValueError("--nearest-pool must be >= 1")
    if max_lines_per_point < 1:
        raise ValueError("--max-lines-per-point must be >= 1")
    if not (0.0 <= reject_angle_low <= 90.0):
        raise ValueError("--reject-angle-low must be in [0, 90]")
    if not (0.0 <= reject_angle_high <= 90.0):
        raise ValueError("--reject-angle-high must be in [0, 90]")
    if reject_angle_low > reject_angle_high:
        raise ValueError("--reject-angle-low must be <= --reject-angle-high")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = BreadboardPinDetector()
    result, debug = detector.detect(image)

    lines = detector.build_neighbor_lines(
        result,
        nearest_pool=nearest_pool,
        max_lines_per_point=max_lines_per_point,
        reject_angle_low=reject_angle_low,
        reject_angle_high=reject_angle_high,
    )

    result["neighbor_line_count"] = int(len(lines))
    result["neighbor_lines"] = lines

    text = json.dumps(result, indent=2)
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        if do_print:
            print(text)

    if debug_image_path:
        dbg = detector.draw_debug(
            image,
            result,
            lines=lines,
            draw_labels=draw_labels,
            dot_radius=dot_radius,
            line_thickness=line_thickness,
        )
        save_debug_panels(debug_image_path, dbg, debug, save_other=save_other_files)
        print(debug_image_path)

    return debug_image_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--json", default="", help="Where to save JSON output")
    parser.add_argument("--debug-image", default="", help="Where to save debug visualization")
    parser.add_argument("--draw-labels", action="store_true", help="Draw row,col labels next to cyan dots")

    parser.add_argument(
        "--nearest-pool",
        type=int,
        default=15,
        help="How many nearest neighbors to inspect per cyan point",
    )
    parser.add_argument(
        "--max-lines-per-point",
        type=int,
        default=10,
        help="Maximum accepted non-diagonal lines to draw from each cyan point",
    )
    parser.add_argument(
        "--reject-angle-low",
        type=float,
        default=30.0,
        help="Lower bound of rejected diagonal angle range in degrees",
    )
    parser.add_argument(
        "--reject-angle-high",
        type=float,
        default=60.0,
        help="Upper bound of rejected diagonal angle range in degrees",
    )
    parser.add_argument(
        "--dot-radius",
        type=int,
        default=2,
        help="Radius of cyan dots in visualization",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=1,
        help="Thickness of cyan neighbor lines in visualization",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Enable standard JSON printing (disabled by default)",
    )
    parser.add_argument(
        "--save-other-files",
        action="store_true",
        help="Save additional debug files alongside the main debug image",
    )

    args = parser.parse_args()

    run_detector(
        image_path=args.image,
        json_path=args.json,
        debug_image_path=args.debug_image,
        draw_labels=args.draw_labels,
        nearest_pool=args.nearest_pool,
        max_lines_per_point=args.max_lines_per_point,
        reject_angle_low=args.reject_angle_low,
        reject_angle_high=args.reject_angle_high,
        dot_radius=args.dot_radius,
        line_thickness=args.line_thickness,
        do_print=args.print,
        save_other_files=args.save_other_files
    )


if __name__ == "__main__":
    main()