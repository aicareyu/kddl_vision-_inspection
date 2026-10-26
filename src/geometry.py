"""Geometric primitives for metrology (pixels unless noted)."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Center-to-center Euclidean distance."""
    return float(math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])))


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two orientations in degrees."""
    diff = abs(float(a) - float(b)) % 180.0
    return float(min(diff, 180.0 - diff))


def order_rect_dimensions(width: float, height: float) -> tuple[float, float]:
    """Map a rotated rectangle to (transverse_width, longitudinal_height).

    For a watch-hand leg the longitudinal dimension is the longer side.
    """
    w = abs(float(width))
    h = abs(float(height))
    return (min(w, h), max(w, h))


def min_area_rect_features(contour: np.ndarray) -> dict[str, float | np.ndarray]:
    """Extract orientation-aware box features from a contour."""
    if contour is None or len(contour) < 3:
        raise ValueError("Contour needs at least 3 points for minAreaRect.")
    rect = cv2.minAreaRect(contour.astype(np.float32))
    (cx, cy), (rw, rh), angle = rect
    width, height = order_rect_dimensions(rw, rh)
    box = cv2.boxPoints(rect).astype(np.float32)
    return {
        "center": np.array([cx, cy], dtype=np.float32),
        "width": float(width),
        "height": float(height),
        "angle": float(angle),
        "box": box,
        "raw_size": (float(rw), float(rh)),
        "rect": rect,
    }


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w / 2.0, y + h / 2.0)
    return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))


def contour_solidity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def offset_contour(contour: np.ndarray, dx: float, dy: float) -> np.ndarray:
    shifted = contour.astype(np.float32).copy()
    shifted[:, 0, 0] += float(dx)
    shifted[:, 0, 1] += float(dy)
    return shifted


def contour_to_polygon(contour: np.ndarray) -> Polygon:
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("Need at least 3 points to build a polygon.")
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "GeometryCollection":
            polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
            if not polys:
                raise ValueError("Contour did not produce a valid polygon.")
            poly = max(polys, key=lambda g: g.area)
        elif poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def min_edge_distance(contour_a: np.ndarray, contour_b: np.ndarray) -> float:
    """Minimum edge-to-edge distance between two contours (0 if they touch)."""
    poly_a = contour_to_polygon(contour_a)
    poly_b = contour_to_polygon(contour_b)
    return float(poly_a.distance(poly_b))


def clip_roi(x: int, y: int, w: int, h: int, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Clamp an axis-aligned ROI to the image."""
    ih, iw = image_shape[:2]
    x = int(max(0, min(x, iw - 1)))
    y = int(max(0, min(y, ih - 1)))
    w = int(max(1, min(w, iw - x)))
    h = int(max(1, min(h, ih - y)))
    return x, y, w, h


def extract_roi(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = clip_roi(x, y, w, h, image.shape)
    return image[y : y + h, x : x + w].copy(), (x, y, w, h)


def line_from_points(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit direction vector and a point on the line."""
    p1 = np.asarray(p1, dtype=np.float64).reshape(2)
    p2 = np.asarray(p2, dtype=np.float64).reshape(2)
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        raise ValueError("Degenerate line: points coincide.")
    return direction / norm, p1


def point_line_signed_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Signed perpendicular distance from point to line (origin + t * direction)."""
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    return float(np.dot(np.asarray(point, dtype=np.float64) - origin, normal))


def fit_line_lsq(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line fit. Returns (unit direction, centroid)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to fit a line.")
    centroid = pts.mean(axis=0)
    uu, ss, vt = np.linalg.svd(pts - centroid)
    direction = vt[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        raise ValueError("Line fit failed.")
    return direction / norm, centroid


def parallel_line_distance(dir_a: np.ndarray, origin_a: np.ndarray, origin_b: np.ndarray) -> float:
    """Distance between two parallel lines sharing ``dir_a``."""
    return abs(point_line_signed_distance(origin_b, origin_a, dir_a))


def box_side_pairs(box: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return ((long_side_p1, long_side_p2), (short_side_p1, short_side_p2))."""
    pts = np.asarray(box, dtype=np.float64).reshape(4, 2)
    edges = [(pts[i], pts[(i + 1) % 4], float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))) for i in range(4)]
    edges_sorted = sorted(edges, key=lambda e: e[2], reverse=True)
    long_side = (edges_sorted[0][0], edges_sorted[0][1])
    short_side = (edges_sorted[-1][0], edges_sorted[-1][1])
    return long_side, short_side
