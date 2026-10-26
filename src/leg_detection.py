"""Contour-based detection and two-leg identification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from .geometry import (
    angular_difference_deg,
    contour_centroid,
    contour_solidity,
    euclidean_distance,
    min_area_rect_features,
    offset_contour,
)

logger = logging.getLogger(__name__)

TWO_LEGS_NOT_DETECTED = "INSPECTION FAILED — TWO LEGS NOT DETECTED"
MULTIPLE_CANDIDATES = "MULTIPLE LEG CANDIDATES DETECTED — REVIEW REQUIRED"


@dataclass
class LegCandidate:
    """A single connected component that may be a watch-hand leg."""

    contour: np.ndarray
    area: float
    bbox: tuple[int, int, int, int]
    width: float
    height: float
    aspect_ratio: float
    centroid: tuple[float, float]
    orientation: float
    rotated_box: np.ndarray
    perimeter: float
    solidity: float
    rect: tuple

    def features(self) -> dict[str, float]:
        return {
            "area": self.area,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "orientation": self.orientation,
            "perimeter": self.perimeter,
            "solidity": self.solidity,
            "cx": self.centroid[0],
            "cy": self.centroid[1],
        }


@dataclass
class PairResult:
    status: str
    message: str
    leg1: Optional[LegCandidate] = None
    leg2: Optional[LegCandidate] = None
    candidates: list[LegCandidate] = field(default_factory=list)
    score: float = 0.0


def _in_range(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def _passes_filters(candidate: LegCandidate, cfg: dict[str, Any]) -> bool:
    ar_range = cfg.get("aspect_ratio_range", [0.15, 4.0])
    lo_ar, hi_ar = float(ar_range[0]), float(ar_range[1])
    checks = [
        _in_range(candidate.area, float(cfg.get("min_area", 0)), float(cfg.get("max_area", 1e12))),
        _in_range(candidate.width, float(cfg.get("min_width", 0)), float(cfg.get("max_width", 1e12))),
        _in_range(candidate.height, float(cfg.get("min_height", 0)), float(cfg.get("max_height", 1e12))),
        _in_range(candidate.aspect_ratio, lo_ar, hi_ar),
        candidate.solidity >= float(cfg.get("min_solidity", 0.0)),
    ]
    return all(checks)


def contour_to_candidate(contour: np.ndarray, offset: tuple[int, int] = (0, 0)) -> Optional[LegCandidate]:
    if contour is None or len(contour) < 3:
        return None
    shifted = offset_contour(contour, offset[0], offset[1])
    area = float(cv2.contourArea(shifted))
    if area < 1:
        return None
    try:
        feats = min_area_rect_features(shifted)
    except (cv2.error, ValueError):
        return None
    width = float(feats["width"])
    height = float(feats["height"])
    if height <= 0:
        return None
    x, y, bw, bh = cv2.boundingRect(shifted.astype(np.int32))
    return LegCandidate(
        contour=shifted,
        area=area,
        bbox=(int(x), int(y), int(bw), int(bh)),
        width=width,
        height=height,
        aspect_ratio=float(width / height),
        centroid=contour_centroid(shifted),
        orientation=float(feats["angle"]),
        rotated_box=np.asarray(feats["box"]),
        perimeter=float(cv2.arcLength(shifted, True)),
        solidity=contour_solidity(shifted),
        rect=feats["rect"],
    )


def extract_candidates(
    binary: np.ndarray,
    config: dict[str, Any],
    offset: tuple[int, int] = (0, 0),
) -> list[LegCandidate]:
    """Find and filter leg-like connected components."""
    det_cfg = config.get("leg_detection", {})
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[LegCandidate] = []
    for contour in contours:
        cand = contour_to_candidate(contour, offset)
        if cand is None:
            continue
        if _passes_filters(cand, det_cfg):
            candidates.append(cand)
    candidates.sort(key=lambda c: c.area, reverse=True)
    logger.info("Leg candidates after filtering: %d / %d contours", len(candidates), len(contours))
    return candidates


def _largest_component_mask(binary: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    plate = max(contours, key=cv2.contourArea)
    if cv2.contourArea(plate) < 200:
        return None
    mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [plate], -1, 255, -1)
    return mask


def _pin_body_mask(
    gray: np.ndarray,
    body_mask: Optional[np.ndarray],
    color: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Whole-plate silhouette. Prefer brass color so textured plates stay one blob."""
    ranked: list[tuple[float, np.ndarray]] = []
    if color is not None and color.ndim == 3:
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        brass = cv2.inRange(hsv, (5, 30, 60), (40, 255, 255))
        brass = cv2.morphologyEx(brass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        filled = _largest_component_mask(brass)
        if filled is not None:
            ranked.append((float((filled > 0).mean()), filled))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    filled = _largest_component_mask(inv)
    if filled is not None:
        ranked.append((float((filled > 0).mean()), filled))
    mask = body_mask
    if mask is not None and mask.size == gray.size:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        filled = _largest_component_mask(mask)
        if filled is not None:
            ranked.append((float((filled > 0).mean()), filled))
    usable = [(fill, item) for fill, item in ranked if 0.04 <= fill <= 0.80]
    if usable:
        usable.sort(key=lambda pair: pair[0], reverse=True)
        return usable[0][1]
    if ranked:
        return max(ranked, key=lambda pair: pair[0])[1]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fill = float((otsu > 0).mean())
    return otsu if 0.01 < fill < 0.55 else cv2.bitwise_not(otsu)


def _order_box_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [pts[int(np.argmin(sums))], pts[int(np.argmin(diffs))], pts[int(np.argmax(sums))], pts[int(np.argmax(diffs))]],
        dtype=np.float32,
    )


def _circle_candidate(cx: float, cy: float, radius: float, offset: tuple[int, int]) -> Optional[LegCandidate]:
    radius = max(2.5, float(radius))
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    pts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    contour = pts.reshape(-1, 1, 2).astype(np.float32)
    return contour_to_candidate(contour, offset)


def _pin_filters(pin_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_area": float(pin_cfg.get("min_area", 6)),
        "max_area": float(pin_cfg.get("max_area", 1200)),
        "min_width": float(pin_cfg.get("min_width", 2)),
        "max_width": float(pin_cfg.get("max_width", 50)),
        "min_height": float(pin_cfg.get("min_height", 2)),
        "max_height": float(pin_cfg.get("max_height", 90)),
        "aspect_ratio_range": pin_cfg.get("aspect_ratio_range", [0.15, 4.0]),
        "min_solidity": float(pin_cfg.get("min_solidity", 0.30)),
    }


def _candidates_from_binary(
    binary: np.ndarray,
    filters: dict[str, Any],
    offset: tuple[int, int],
) -> list[LegCandidate]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found: list[LegCandidate] = []
    for contour in contours:
        cand = contour_to_candidate(contour, offset)
        if cand is not None and _passes_filters(cand, filters):
            found.append(cand)
    return found


def _strongest_peak(score: np.ndarray, mask: np.ndarray) -> Optional[tuple[int, int]]:
    if mask is None or int((mask > 0).sum()) < 10:
        return None
    work = score.astype(np.float32).copy()
    work[mask == 0] = 0
    if float(work.max()) < 1.0:
        return None
    _min, _max, _min_loc, max_loc = cv2.minMaxLoc(work)
    return int(max_loc[0]), int(max_loc[1])


def _pin_score(gray: np.ndarray) -> np.ndarray:
    """Small isolated pin contrast; long plate edges are suppressed."""
    k = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 15))
    contrast = cv2.max(
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
    )
    score = cv2.GaussianBlur(cv2.max(residual, contrast), (5, 5), 0)
    score = cv2.subtract(
        score,
        cv2.morphologyEx(score, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))),
    )
    edges = cv2.Canny(gray, 35, 110)
    long_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1)))
    long_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17)))
    rim = cv2.dilate(cv2.max(long_h, long_v), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    score = score.astype(np.float32)
    score[rim > 0] *= 0.15
    return cv2.GaussianBlur(score, (5, 5), 0)


def _peaks_left_right(score: np.ndarray, interior: np.ndarray, min_sep: float) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    ys, xs = np.where(interior > 0)
    if xs.size < 20:
        return None
    mid_x = int(round(float(xs.min() + xs.max()) / 2.0))
    left_mask = interior.copy()
    right_mask = interior.copy()
    left_mask[:, mid_x:] = 0
    right_mask[:, :mid_x] = 0
    left = _strongest_peak(score, left_mask)
    right = _strongest_peak(score, right_mask)
    if left is None or right is None:
        return None
    if float(np.hypot(right[0] - left[0], right[1] - left[1])) < min_sep:
        return None
    return left, right


def _extract_pins_warped(
    gray: np.ndarray,
    plate_mask: np.ndarray,
    offset: tuple[int, int],
) -> list[LegCandidate]:
    """Warp the plate to a rectangle and pick one pin in each half.

    The photographed 3D corner sits inside the raw silhouette (side wall
    below it). After a min-area-rect warp that corner is on the rim again,
    so Inter-Leg Distance (D) is pin-to-pin.
    """
    if plate_mask is None or int((plate_mask > 0).sum()) < 200:
        return []
    contours, _ = cv2.findContours((plate_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    plate = max(contours, key=cv2.contourArea)
    if cv2.contourArea(plate) < 200:
        return []
    rect = cv2.minAreaRect(plate.astype(np.float32))
    box = _order_box_points(cv2.boxPoints(rect))
    width = float(max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3]), 1.0))
    height = float(max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1]), 1.0))
    if width < height:
        box = np.array([box[3], box[0], box[1], box[2]], dtype=np.float32)
        width, height = height, width
    dst_w = max(48, int(round(width)))
    dst_h = max(24, int(round(height)))
    dst = np.array(
        [[0.0, 0.0], [dst_w - 1.0, 0.0], [dst_w - 1.0, dst_h - 1.0], [0.0, dst_h - 1.0]],
        dtype=np.float32,
    )
    warp = cv2.getPerspectiveTransform(box, dst)
    inverse = cv2.getPerspectiveTransform(dst, box)
    warped = cv2.warpPerspective(gray, warp, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderValue=255)
    warped_mask = cv2.warpPerspective(plate_mask, warp, (dst_w, dst_h), flags=cv2.INTER_NEAREST)

    # Drop the side wall (lower strip) and the rectangle rim / 3D corners.
    face = warped_mask.copy()
    cut = max(4, int(0.22 * dst_h))
    face[dst_h - cut :, :] = 0
    rim_x = max(4, int(0.10 * dst_w))
    rim_y = max(3, int(0.08 * dst_h))
    face[:rim_y, :] = 0
    face[:, :rim_x] = 0
    face[:, dst_w - rim_x :] = 0
    dist = cv2.distanceTransform((face > 0).astype(np.uint8), cv2.DIST_L2, 5)
    interior = ((dist > max(3.0, 0.12 * float(dist.max() or 1.0))) & (face > 0)).astype(np.uint8) * 255
    if int((interior > 0).sum()) < 80:
        interior = face
    if int((interior > 0).sum()) < 40:
        return []

    score = _pin_score(warped)
    score[interior == 0] = 0
    if float(score.max()) < 3:
        return []
    pair = _peaks_left_right(score, interior, min_sep=max(16.0, 0.28 * dst_w))
    if pair is None:
        return []

    radius = max(3.5, 0.014 * width)
    out: list[LegCandidate] = []
    for px, py in pair:
        mapped = inverse @ np.array([float(px), float(py), 1.0], dtype=np.float64)
        x, y = float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])
        ix, iy = int(round(x)), int(round(y))
        if iy < 0 or ix < 0 or iy >= plate_mask.shape[0] or ix >= plate_mask.shape[1]:
            continue
        if plate_mask[iy, ix] == 0:
            continue
        cand = _circle_candidate(x, y, radius, offset)
        if cand is not None:
            out.append(cand)
    return out


def _two_interior_pins(
    gray: np.ndarray,
    plate_mask: np.ndarray,
    offset: tuple[int, int],
) -> list[LegCandidate]:
    """Fallback: two far-apart peaks inside the plate silhouette."""
    if plate_mask is None or int((plate_mask > 0).sum()) < 200:
        return []
    dist = cv2.distanceTransform((plate_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) < 6:
        return []
    interior = ((dist > max(6.0, 0.18 * float(dist.max()))) & (plate_mask > 0)).astype(np.uint8) * 255
    if int((interior > 0).sum()) < 80:
        interior = ((dist > max(3.0, 0.10 * float(dist.max()))) & (plate_mask > 0)).astype(np.uint8) * 255
    score = _pin_score(gray)
    score[interior == 0] = 0
    if float(score.max()) < 3:
        return []
    ys, xs = np.where(interior > 0)
    if xs.size == 0:
        return []
    plate_len = float(max(xs.max() - xs.min(), ys.max() - ys.min(), 1))
    pair = _peaks_left_right(score, interior, min_sep=max(18.0, 0.28 * plate_len))
    if pair is None:
        return []
    radius = max(3.5, 0.012 * plate_len)
    out: list[LegCandidate] = []
    for x, y in pair:
        cand = _circle_candidate(float(x), float(y), radius, offset)
        if cand is not None:
            out.append(cand)
    return out


def _extract_pins_aligned(
    gray: np.ndarray,
    plate: np.ndarray,
    pin_cfg: dict[str, Any],
    offset: tuple[int, int],
) -> list[LegCandidate]:
    """Find one pin near each end of the plate (front / back)."""
    if cv2.contourArea(plate) < 200:
        return []
    plate_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.drawContours(plate_mask, [plate.astype(np.int32)], -1, 255, -1)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(plate.astype(np.float32))
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    img_h, img_w = gray.shape[:2]
    rotate = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)
    rot = cv2.warpAffine(gray, rotate, (img_w, img_h), flags=cv2.INTER_LINEAR, borderValue=255)
    rot_mask = cv2.warpAffine(plate_mask, rotate, (img_w, img_h), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rot_mask > 0)
    if xs.size < 50:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = rot[y0 : y1 + 1, x0 : x1 + 1]
    crop_mask = rot_mask[y0 : y1 + 1, x0 : x1 + 1]
    ch, cw = crop.shape[:2]
    if ch < 12 or cw < 24:
        return []
    rim = max(4, int(0.06 * min(ch, cw)))
    inner = cv2.erode(crop_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim, rim)))
    if int((inner > 0).sum()) < 40:
        inner = crop_mask
    k = int(pin_cfg.get("tophat_kernel", 15))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    residual = cv2.absdiff(crop, cv2.medianBlur(crop, min(k, 15) | 1))
    contrast = cv2.max(
        cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, kernel),
        cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel),
    )
    score = cv2.GaussianBlur(cv2.max(residual, contrast), (5, 5), 0)
    score[inner == 0] = 0
    if float(score.max()) < 4:
        return []

    mid = cw // 2
    left_mask = inner.copy()
    right_mask = inner.copy()
    left_mask[:, mid:] = 0
    right_mask[:, :mid] = 0
    edge = max(3, int(0.08 * cw))
    left_mask[:, :edge] = 0
    right_mask[:, cw - edge :] = 0
    left = _strongest_peak(score, left_mask)
    right = _strongest_peak(score, right_mask)
    if left is None or right is None:
        return []

    inverse = cv2.invertAffineTransform(rotate)
    radius = max(3.5, 0.016 * float(rw))
    candidates: list[LegCandidate] = []
    for px, py in (left, right):
        col = float(px + x0)
        row = float(py + y0)
        mapped = inverse @ np.array([col, row, 1.0], dtype=np.float64)
        ix, iy = int(round(mapped[0])), int(round(mapped[1]))
        if iy < 0 or ix < 0 or iy >= plate_mask.shape[0] or ix >= plate_mask.shape[1]:
            continue
        if plate_mask[iy, ix] == 0:
            continue
        cand = _circle_candidate(float(mapped[0]), float(mapped[1]), radius, offset)
        if cand is not None:
            candidates.append(cand)
    return candidates


def extract_pin_candidates(
    gray: np.ndarray,
    body_mask: np.ndarray,
    config: dict[str, Any],
    offset: tuple[int, int] = (0, 0),
    color: Optional[np.ndarray] = None,
) -> list[LegCandidate]:
    """Find two small pins that merge into the body under normal thresholding."""
    pin_cfg = config.get("leg_detection", {}).get("pin_mode", {})
    if not pin_cfg.get("enabled", True):
        return []
    if gray is None or gray.size == 0:
        return []
    if gray.ndim == 3:
        color = gray
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    body = _pin_body_mask(gray, body_mask, color=color)
    warped_pins = _extract_pins_warped(gray, body, offset)
    if len(warped_pins) >= 2:
        logger.info("Pin candidates from warped top-face search: %d", len(warped_pins))
        return warped_pins
    interior_pins = _two_interior_pins(gray, body, offset)
    if len(interior_pins) >= 2:
        logger.info("Pin candidates from interior plate search: %d", len(interior_pins))
        return interior_pins
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aligned: list[LegCandidate] = []
    if contours:
        plate = max(contours, key=cv2.contourArea)
        aligned = _extract_pins_aligned(gray, plate, pin_cfg, offset)
        if len(aligned) >= 2:
            logger.info("Pin candidates from aligned plate search: %d", len(aligned))
            return aligned

    erode = int(pin_cfg.get("body_erode", 13))
    if erode < 1:
        erode = 1
    inner = cv2.erode(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
    if int((inner > 0).sum()) < 80:
        inner = body

    k = int(pin_cfg.get("tophat_kernel", 21))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, min(k, 15) | 1))
    contrast = cv2.max(
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
    )
    enhanced = cv2.max(residual, contrast)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=inner)
    if int(enhanced.max()) < 4:
        return aligned
    plate_vals = gray[inner > 0]
    if plate_vals.size < 50:
        return aligned
    med = float(np.median(plate_vals))
    dark = ((gray < (med - float(np.clip(med * 0.22, 10.0, 45.0)))) & (inner > 0)).astype(np.uint8) * 255
    bright = ((gray > (med + float(np.clip((255.0 - med) * 0.18, 10.0, 45.0)))) & (inner > 0)).astype(np.uint8) * 255
    cthr = max(8.0, float(np.percentile(enhanced[inner > 0], float(pin_cfg.get("contrast_percentile", 99.4)))))
    hot = ((enhanced >= cthr) & (inner > 0)).astype(np.uint8) * 255
    binary = cv2.bitwise_or(cv2.bitwise_or(dark, bright), hot)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    found = _candidates_from_binary(binary, _pin_filters(pin_cfg), offset)
    found.sort(key=lambda c: c.area, reverse=True)
    logger.info("Pin candidates after filtering: %d", len(found))
    return (found or aligned)[:12]


def _size_similarity(a: LegCandidate, b: LegCandidate) -> float:
    denom = max(a.area, b.area, 1e-6)
    return float(1.0 - abs(a.area - b.area) / denom)


def _shape_similarity(a: LegCandidate, b: LegCandidate) -> float:
    denom = max(a.aspect_ratio, b.aspect_ratio, 1e-6)
    ar = 1.0 - abs(a.aspect_ratio - b.aspect_ratio) / denom
    wh = 1.0 - abs(a.width - b.width) / max(a.width, b.width, 1e-6)
    hh = 1.0 - abs(a.height - b.height) / max(a.height, b.height, 1e-6)
    return float(np.clip((ar + wh + hh) / 3.0, 0.0, 1.0))


def _orientation_similarity(a: LegCandidate, b: LegCandidate, max_diff: float) -> float:
    diff = angular_difference_deg(a.orientation, b.orientation)
    if diff > max_diff:
        return 0.0
    return float(1.0 - diff / max(max_diff, 1e-6))


def _distance_score(dist: float, dmin: float, dmax: float) -> float:
    if dmin <= dist <= dmax:
        return 1.0
    if dist <= 0:
        return 0.0
    if dist < dmin:
        return float(np.clip(dist / max(dmin, 1e-6), 0.0, 1.0))
    span = max(dmax, 1.0)
    return float(np.clip(1.0 - (dist - dmax) / span, 0.0, 1.0))


def _position_score(a: LegCandidate, b: LegCandidate, max_axis_ratio: float) -> float:
    dx = abs(a.centroid[0] - b.centroid[0])
    dy = abs(a.centroid[1] - b.centroid[1])
    major = max(dx, dy, 1e-6)
    minor = min(dx, dy)
    ratio = major / max(minor, 1e-6)
    # Prefer pairs that share a row or a column (typical two-leg layout).
    return float(np.clip(ratio / max(max_axis_ratio, 1.0), 0.0, 1.0)) if ratio >= 1.5 else 0.4


def _unique_best_pair(
    candidates: list[LegCandidate],
    pair_cfg: dict[str, Any],
    min_score: float,
    min_gap: float,
) -> Optional[tuple[float, LegCandidate, LegCandidate]]:
    """Return the top pair only when it is clearly better than the runner-up."""
    scored: list[tuple[float, LegCandidate, LegCandidate]] = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            scored.append((score_pair(first, second, pair_cfg), first, second))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0]
    if best[0] < min_score:
        return None
    if len(scored) > 1 and (best[0] - scored[1][0]) < min_gap:
        return None
    return best


def _best_pin_pair(
    candidates: list[LegCandidate],
    pair_cfg: dict[str, Any],
    min_score: float,
    min_gap: float,
) -> Optional[tuple[float, LegCandidate, LegCandidate]]:
    """Pick two similar, well-separated pins when extra specks are present."""
    unique = _unique_best_pair(candidates, pair_cfg, min_score, min_gap)
    if unique is not None:
        return unique
    dmin = float(pair_cfg.get("expected_distance_px_min", 20))
    dmax = float(pair_cfg.get("expected_distance_px_max", 1200))
    ranked: list[tuple[float, float, float, LegCandidate, LegCandidate]] = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            dist = euclidean_distance(first.centroid, second.centroid)
            if dist < dmin or dist > dmax:
                continue
            similar = 0.6 * _size_similarity(first, second) + 0.4 * _shape_similarity(first, second)
            if similar < 0.55:
                continue
            ranked.append((similar, dist, score_pair(first, second, pair_cfg), first, second))
    if not ranked:
        return None
    # Prefer similar pins that are far apart (one pin each end of the plate).
    span = max(item[1] for item in ranked) or 1.0
    ranked.sort(key=lambda item: (item[0] * (0.45 + 0.55 * item[1] / span), item[1]), reverse=True)
    similar, dist, score, first, second = ranked[0]
    if dist < dmin:
        return None
    if score < min_score * 0.70 and similar < 0.70:
        return None
    return score, first, second


def score_pair(a: LegCandidate, b: LegCandidate, pair_cfg: dict[str, Any]) -> float:
    dist = euclidean_distance(a.centroid, b.centroid)
    size_s = _size_similarity(a, b)
    ori_s = _orientation_similarity(a, b, float(pair_cfg.get("max_orientation_diff_deg", 35)))
    dist_s = _distance_score(
        dist,
        float(pair_cfg.get("expected_distance_px_min", 10)),
        float(pair_cfg.get("expected_distance_px_max", 400)),
    )
    pos_s = _position_score(a, b, float(pair_cfg.get("max_centroid_axis_ratio", 8.0)))
    shape_s = _shape_similarity(a, b)
    score = (
        float(pair_cfg.get("size_weight", 0.25)) * size_s
        + float(pair_cfg.get("orientation_weight", 0.20)) * ori_s
        + float(pair_cfg.get("distance_weight", 0.20)) * dist_s
        + float(pair_cfg.get("position_weight", 0.15)) * pos_s
        + float(pair_cfg.get("shape_weight", 0.20)) * shape_s
    )
    return float(np.clip(score, 0.0, 1.0))


def _order_legs(a: LegCandidate, b: LegCandidate) -> tuple[LegCandidate, LegCandidate]:
    """Leg 1 is the left-most (then top-most) candidate."""
    if a.centroid[0] < b.centroid[0] - 1e-6:
        return a, b
    if b.centroid[0] < a.centroid[0] - 1e-6:
        return b, a
    return (a, b) if a.centroid[1] <= b.centroid[1] else (b, a)


def identify_two_legs(
    candidates: list[LegCandidate],
    config: dict[str, Any],
    *,
    allow_best_pair: bool = False,
) -> PairResult:
    """Return exactly two legs, or a review / failure status.

    Contour lists with more than two items stay REVIEW. Pin search may pass
    ``allow_best_pair=True`` to take the most similar, well-separated pair.
    """
    pair_cfg = config.get("leg_detection", {}).get("pairing", {})
    min_score = float(pair_cfg.get("min_score", 0.45))

    if len(candidates) < 2:
        logger.warning("Two-leg identification failed: %d candidate(s)", len(candidates))
        return PairResult(status="FAILED", message=TWO_LEGS_NOT_DETECTED, candidates=candidates)

    if len(candidates) > 2:
        pin_cfg = config.get("leg_detection", {}).get("pin_mode", {})
        if allow_best_pair:
            picked = _best_pin_pair(candidates, pair_cfg, min_score, float(pin_cfg.get("pair_score_gap", 0.04)))
            if picked is not None:
                score, a, b = picked
                leg1, leg2 = _order_legs(a, b)
                return PairResult(
                    status="OK",
                    message="Two legs identified (best pin pair).",
                    leg1=leg1,
                    leg2=leg2,
                    candidates=candidates,
                    score=score,
                )
        logger.warning("Two-leg identification requires review: %d candidates", len(candidates))
        return PairResult(status="REVIEW", message=MULTIPLE_CANDIDATES, candidates=candidates)

    a, b = candidates[0], candidates[1]
    score = score_pair(a, b, pair_cfg)
    dist = euclidean_distance(a.centroid, b.centroid)
    if allow_best_pair and dist >= float(pair_cfg.get("expected_distance_px_min", 20)):
        score = max(score, min_score)
    if score < min_score:
        logger.warning("Pair score %.3f below threshold %.3f", score, min_score)
        return PairResult(
            status="FAILED",
            message=TWO_LEGS_NOT_DETECTED + f" (pair score {score:.2f} < {min_score:.2f})",
            candidates=candidates,
            score=score,
        )
    leg1, leg2 = _order_legs(a, b)
    return PairResult(
        status="OK",
        message="Two legs identified.",
        leg1=leg1,
        leg2=leg2,
        candidates=candidates,
        score=score,
    )


def draw_candidates(image: np.ndarray, candidates: list[LegCandidate]) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    box_color = (0, 180, 255)
    for i, cand in enumerate(candidates, start=1):
        cv2.drawContours(canvas, [cand.contour.astype(np.int32)], -1, box_color, 2)
        x, y, w, h = cand.bbox
        pad = 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = x + w + pad, y + h + pad
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), 5)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), box_color, 2)
        if cand.rotated_box is not None and len(cand.rotated_box) >= 3:
            pts = cand.rotated_box.astype(np.int32).reshape(-1, 2)
            cv2.polylines(canvas, [pts], True, box_color, 2, cv2.LINE_AA)
        cx, cy = int(cand.centroid[0]), int(cand.centroid[1])
        cv2.circle(canvas, (cx, cy), 5, box_color, -1, cv2.LINE_AA)
        label = f"C{i}  W {cand.width:.1f}px"
        cv2.putText(canvas, label, (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
    return canvas
