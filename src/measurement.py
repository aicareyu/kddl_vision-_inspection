"""Orientation-aware dimensioning, optional sub-pixel refinement, inspection run."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from .calibration import CalibrationError, CalibrationState, apply_camera_undistort, require_production_calibration
from .geometry import (
    box_side_pairs,
    clip_roi,
    euclidean_distance,
    extract_roi,
    fit_line_lsq,
    min_area_rect_features,
    min_edge_distance,
    order_rect_dimensions,
    parallel_line_distance,
)
from .leg_detection import (
    LegCandidate,
    PairResult,
    contour_to_candidate,
    draw_candidates,
    extract_candidates,
    extract_pin_candidates,
    identify_two_legs,
)
from .preprocessing import preprocess, to_gray
from .segmentation import segment
from .validation import ToleranceResult, evaluate_object_tolerances, evaluate_tolerances

logger = logging.getLogger(__name__)


@dataclass
class LegMeasurement:
    label: str
    width_px: float
    height_px: float
    width_mm: Optional[float]
    height_mm: Optional[float]
    angle_deg: float
    centroid: tuple[float, float]
    contour: np.ndarray
    box_points: np.ndarray
    subpixel_used: bool = False
    tip: Optional[tuple[float, float]] = None
    base: Optional[tuple[float, float]] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "angle_deg": self.angle_deg,
            "centroid": self.centroid,
            "subpixel_used": self.subpixel_used,
        }


@dataclass
class InspectionResult:
    status: str
    message: str
    overall_result: str = "NG"
    leg1: Optional[LegMeasurement] = None
    leg2: Optional[LegMeasurement] = None
    center_distance_px: Optional[float] = None
    center_distance_mm: Optional[float] = None
    edge_distance_px: Optional[float] = None
    edge_distance_mm: Optional[float] = None
    overall_width_px: Optional[float] = None
    overall_height_px: Optional[float] = None
    overall_width_mm: Optional[float] = None
    overall_height_mm: Optional[float] = None
    pair_score: float = 0.0
    confidence: float = 0.0
    calibration: CalibrationState = field(default_factory=CalibrationState)
    tolerance: Optional[ToleranceResult] = None
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    overlay: Optional[np.ndarray] = None
    image_name: str = ""
    roi: Optional[tuple[int, int, int, int]] = None
    candidate_count: int = 0
    mode: str = "object"
    object: Optional[LegMeasurement] = None
    object_area_px: Optional[float] = None
    object_area_mm2: Optional[float] = None


def pixels_to_mm(pixels: float, mm_per_pixel: float) -> float:
    if mm_per_pixel <= 0:
        raise CalibrationError("mm_per_pixel must be > 0.")
    return float(pixels) * float(mm_per_pixel)


def _maybe_mm(pixels: Optional[float], state: CalibrationState) -> Optional[float]:
    if pixels is None:
        return None
    if not state.is_production_ready:
        return None
    return state.pixels_to_mm(pixels)


def _collect_edge_points_near_line(
    contour: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float64)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return np.empty((0, 2))
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    rel = pts - p1
    along = rel @ direction
    dist = np.abs(rel @ normal)
    mask = (along >= -2.0) & (along <= length + 2.0) & (dist <= max_dist)
    return pts[mask]


def refine_leg_subpixel(
    candidate: LegCandidate,
    gray_full: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, float, np.ndarray, bool]:
    """Refine width/height using cornerSubPix + least-squares edge fits."""
    sub_cfg = config.get("measurement", {}).get("subpixel", {})
    if not sub_cfg.get("enabled", False):
        feats = min_area_rect_features(candidate.contour)
        return float(feats["width"]), float(feats["height"]), np.asarray(feats["box"]), False

    box = candidate.rotated_box.astype(np.float32).reshape(-1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(sub_cfg.get("max_iterations", 40)),
        float(sub_cfg.get("epsilon", 0.01)),
    )
    win = int(sub_cfg.get("corner_win_size", 5))
    zero = int(sub_cfg.get("corner_zero_zone", -1))
    try:
        refined = cv2.cornerSubPix(gray_full, box, (win, win), (zero, zero), criteria)
        refined_box = refined.reshape(4, 2)
    except cv2.error:
        refined_box = candidate.rotated_box.reshape(4, 2)

    long_side, short_side = box_side_pairs(refined_box)
    width_px, height_px = order_rect_dimensions(
        float(np.linalg.norm(short_side[1] - short_side[0])),
        float(np.linalg.norm(long_side[1] - long_side[0])),
    )

    # Distance between opposite fitted edges is more stable than corner length.
    try:
        long_pts = _collect_edge_points_near_line(candidate.contour, long_side[0], long_side[1], 4.0)
        # Opposite long side: shift by the short-side vector.
        short_vec = short_side[1] - short_side[0]
        opp_long = (long_side[0] + short_vec, long_side[1] + short_vec)
        opp_pts = _collect_edge_points_near_line(candidate.contour, opp_long[0], opp_long[1], 4.0)
        if len(long_pts) >= 8 and len(opp_pts) >= 8:
            dir_a, origin_a = fit_line_lsq(long_pts)
            _, origin_b = fit_line_lsq(opp_pts)
            width_fit = parallel_line_distance(dir_a, origin_a, origin_b)
            if width_fit > 0:
                width_px = float(width_fit)

        short_pts = _collect_edge_points_near_line(candidate.contour, short_side[0], short_side[1], 4.0)
        long_vec = long_side[1] - long_side[0]
        opp_short = (short_side[0] + long_vec, short_side[1] + long_vec)
        opp_short_pts = _collect_edge_points_near_line(candidate.contour, opp_short[0], opp_short[1], 4.0)
        if len(short_pts) >= 8 and len(opp_short_pts) >= 8:
            dir_s, origin_s = fit_line_lsq(short_pts)
            _, origin_t = fit_line_lsq(opp_short_pts)
            height_fit = parallel_line_distance(dir_s, origin_s, origin_t)
            if height_fit > 0:
                height_px = float(height_fit)
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.debug("Sub-pixel line fit fell back to minAreaRect: %s", exc)

    width_px, height_px = order_rect_dimensions(width_px, height_px)
    return width_px, height_px, refined_box.astype(np.float32), True


def _measure_pin_stem(gray: np.ndarray, cx: float, cy: float) -> Optional[dict[str, Any]]:
    """Tip-to-base height along the dark/high-contrast pin column."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    if not (0 <= ix < img_w and 0 <= iy < img_h):
        return None
    rad = int(max(18, 0.06 * min(img_h, img_w)))
    x0, x1 = max(0, ix - rad), min(img_w, ix + rad + 1)
    y0, y1 = max(0, iy - rad), min(img_h, iy + rad + 1)
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    if crop.size < 25:
        return None
    lx, ly = ix - x0, iy - y0
    med = float(np.median(crop))
    residual = cv2.absdiff(crop.astype(np.uint8), cv2.medianBlur(crop.astype(np.uint8), 11)).astype(np.float32)
    pinness = cv2.GaussianBlur(np.maximum(med - crop, 0.0) + residual, (3, 3), 0)
    # Snap to the strongest pin pixel near the detected centre (handles bright tips).
    y_a, y_b = max(0, ly - 4), min(pinness.shape[0], ly + 5)
    x_a, x_b = max(0, lx - 4), min(pinness.shape[1], lx + 5)
    patch = pinness[y_a:y_b, x_a:x_b]
    if patch.size == 0 or float(patch.max()) < 6.0:
        return None
    py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
    lx, ly = int(x_a + px), int(y_a + py)
    half = 2
    col = pinness[:, max(0, lx - half) : min(pinness.shape[1], lx + half + 1)]
    profile = np.max(col, axis=1)
    peak = float(profile[ly])
    if peak < 8.0:
        return None
    thresh = max(8.0, 0.38 * peak)
    top = ly
    while top > 0 and float(profile[top - 1]) >= thresh:
        top -= 1
    bot = ly
    while bot < profile.size - 1 and float(profile[bot + 1]) >= thresh:
        bot += 1
    height = float(bot - top + 1)
    if height < 5.0:
        return None
    mid = int(round(0.5 * (top + bot)))
    row = pinness[mid, :]
    wthresh = max(6.0, 0.38 * float(row.max()))
    xs = np.where(row >= wthresh)[0]
    width = float(xs.max() - xs.min() + 1) if xs.size >= 2 else 3.0
    width = float(np.clip(width, 2.0, max(3.0, 0.55 * height)))
    return {
        "width": width,
        "height": height,
        "tip": (float(x0 + lx), float(y0 + top)),
        "base": (float(x0 + lx), float(y0 + bot)),
    }


def measure_leg(
    candidate: LegCandidate,
    label: str,
    gray_full: np.ndarray,
    state: CalibrationState,
    config: dict[str, Any],
) -> LegMeasurement:
    width_px, height_px, box, used = refine_leg_subpixel(candidate, gray_full, config)
    tip = base = None
    stem = _measure_pin_stem(gray_full, candidate.centroid[0], candidate.centroid[1])
    short = max(min(candidate.width, candidate.height), 1e-6)
    compact = max(candidate.width, candidate.height) < 1.6 * short
    if stem is not None and compact:
        width_px = float(stem["width"])
        height_px = float(stem["height"])
        tip = stem["tip"]
        base = stem["base"]
    return LegMeasurement(
        label=label,
        width_px=width_px,
        height_px=height_px,
        width_mm=_maybe_mm(width_px, state),
        height_mm=_maybe_mm(height_px, state),
        angle_deg=float(candidate.orientation),
        centroid=candidate.centroid,
        contour=candidate.contour,
        box_points=box,
        subpixel_used=used,
        tip=tip,
        base=base,
    )


def find_main_object(
    binary: np.ndarray,
    config: dict[str, Any],
    offset: tuple[int, int] = (0, 0),
) -> Optional[LegCandidate]:
    """Return the largest plausible object contour in a binary mask."""
    obj_cfg = config.get("object_detection", {})
    min_area = float(obj_cfg.get("min_area", 200))
    max_area = float(obj_cfg.get("max_area", 5e7))
    min_ratio = float(obj_cfg.get("min_area_ratio", 0.002))
    max_ratio = float(obj_cfg.get("max_area_ratio", 0.90))
    roi_area = float(max(1, binary.shape[0] * binary.shape[1]))
    lo = max(min_area, min_ratio * roi_area)
    hi = min(max_area, max_ratio * roi_area)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best: Optional[np.ndarray] = None
    best_area = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < lo or area > hi:
            continue
        if area > best_area:
            best = contour
            best_area = area
    if best is None:
        return None
    return contour_to_candidate(best, offset)


def _plausible_two_legs(pair: PairResult, binary: np.ndarray) -> bool:
    """Reject plate fragments or two blobs that sit on the same pin."""
    if pair.leg1 is None or pair.leg2 is None:
        return False
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    plate = max(contours, key=cv2.contourArea)
    plate_area = float(cv2.contourArea(plate))
    if plate_area <= 0:
        return True
    if pair.leg1.area > 0.12 * plate_area or pair.leg2.area > 0.12 * plate_area:
        return False
    (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(plate.astype(np.float32))
    long_side = max(float(rw), float(rh), 1.0)
    dist = euclidean_distance(pair.leg1.centroid, pair.leg2.centroid)
    if dist < 0.18 * long_side:
        return False
    return True


def _looks_like_separate_legs(pair: PairResult) -> bool:
    """Watch-hand legs are elongated. Compact specks on a plate are not."""
    if pair.leg1 is None or pair.leg2 is None:
        return False
    for leg in (pair.leg1, pair.leg2):
        long_s = max(leg.width, leg.height)
        short_s = max(min(leg.width, leg.height), 1e-6)
        if long_s < 1.8 * short_s:
            return False
    return True


def _largest_body_bbox(
    binary: np.ndarray,
    offset: tuple[int, int],
    legs: list[LegCandidate],
) -> Optional[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Ignore contours that are essentially a detected leg.
    leg_areas = {id(l.contour): l.area for l in legs}
    best = None
    best_area = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        if any(abs(area - la) / max(la, 1.0) < 0.15 for la in leg_areas.values()):
            continue
        if area > best_area:
            best_area = area
            best = contour
    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(best)
    return (x + offset[0], y + offset[1], w, h)


def _failed(
    message: str,
    state: CalibrationState,
    debug: dict[str, np.ndarray],
    quality: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> InspectionResult:
    payload = extra or {}
    return InspectionResult(
        status=str(payload.get("status", "FAILED")),
        message=message,
        overall_result="NG",
        calibration=state,
        debug_images=debug,
        quality=quality,
        candidate_count=int(payload.get("candidate_count", 0)),
        pair_score=float(payload.get("pair_score", 0.0)),
        confidence=0.0,
        overlay=debug.get("final"),
        image_name=str(payload.get("image_name", "")),
        roi=payload.get("roi"),
        mode=str(payload.get("mode", "object")),
    )


def _measure_object(
    work: np.ndarray,
    binary: np.ndarray,
    gray_full: np.ndarray,
    config: dict[str, Any],
    calibration: CalibrationState,
    debug: dict[str, np.ndarray],
    quality: dict[str, Any],
    roi: tuple[int, int, int, int],
    image_name: str,
) -> InspectionResult:
    """Detect the largest object in the ROI and measure width / height."""
    from .visualization import draw_overlay, to_bgr

    x, y, _, _ = roi
    mask = binary
    candidate = find_main_object(mask, config, offset=(x, y))
    if candidate is None and bool(config.get("object_detection", {}).get("auto_invert", True)):
        mask = cv2.bitwise_not(binary)
        debug["threshold_inverted"] = mask
        candidate = find_main_object(mask, config, offset=(x, y))

    contour_canvas = to_bgr(work)
    if candidate is not None:
        cv2.drawContours(contour_canvas, [candidate.contour.astype(np.int32)], -1, (255, 180, 0), 2)
    debug["contours"] = contour_canvas
    debug["candidates"] = draw_candidates(work, [candidate] if candidate is not None else [])

    extra = {
        "candidate_count": 0 if candidate is None else 1,
        "image_name": image_name,
        "roi": roi,
        "mode": "object",
    }
    if candidate is None:
        debug["final"] = debug["candidates"]
        return _failed("INSPECTION FAILED — OBJECT NOT DETECTED", calibration, debug, quality, extra)

    measured = measure_leg(candidate, "OBJECT", gray_full, calibration, config)
    # Object mode reports axis-aligned size (image X/Y), which matches an uploaded photo.
    _bx, _by, bw, bh = cv2.boundingRect(candidate.contour.astype(np.int32))
    measured.width_px = float(bw)
    measured.height_px = float(bh)
    measured.width_mm = _maybe_mm(float(bw), calibration)
    measured.height_mm = _maybe_mm(float(bh), calibration)
    area_px = float(candidate.area)
    area_mm2 = None
    if calibration.is_production_ready:
        area_mm2 = area_px * (calibration.mm_per_pixel ** 2)

    result = InspectionResult(
        status="OK",
        message="Object measured." if calibration.is_production_ready else "Object measured in pixels. Calibrate to report millimetres.",
        overall_result="OK",
        object=measured,
        overall_width_px=measured.width_px,
        overall_height_px=measured.height_px,
        overall_width_mm=measured.width_mm,
        overall_height_mm=measured.height_mm,
        object_area_px=area_px,
        object_area_mm2=area_mm2,
        confidence=1.0,
        calibration=calibration,
        quality=quality,
        debug_images=debug,
        image_name=image_name,
        roi=roi,
        candidate_count=1,
        mode="object",
    )
    result.tolerance = evaluate_object_tolerances(result, config)
    result.overall_result = result.tolerance.overall if result.tolerance else "OK"
    if quality.get("poor"):
        result.message += f" (quality warning: {quality.get('message')})"

    overlay = draw_overlay(work, result)
    result.overlay = overlay
    result.debug_images["final"] = overlay
    logger.info(
        "Object %s | W %.4f %s | H %.4f %s",
        result.overall_result,
        measured.width_mm if measured.width_mm is not None else measured.width_px,
        "mm" if measured.width_mm is not None else "px",
        measured.height_mm if measured.height_mm is not None else measured.height_px,
        "mm" if measured.height_mm is not None else "px",
    )
    return result


def run_inspection(
    image: np.ndarray,
    config: dict[str, Any],
    calibration: CalibrationState,
    roi: Optional[tuple[int, int, int, int]] = None,
    image_name: str = "",
) -> InspectionResult:
    """Full deterministic inspection of one BGR image."""
    from .visualization import draw_overlay, draw_roi, to_bgr

    config = copy.deepcopy(config)
    # Resizing would change pixel scale and break millimetre conversion.
    config.setdefault("preprocessing", {}).setdefault("resize", {})["enabled"] = False

    debug: dict[str, np.ndarray] = {"original": image.copy()}
    quality: dict[str, Any] = {}

    mode = str(config.get("measurement", {}).get("mode", "object")).lower()
    if mode not in {"object", "two_legs"}:
        mode = "object"

    if image is None or image.size == 0:
        return _failed("Image not loaded or is empty.", calibration, debug, quality, {"mode": mode})

    if mode == "two_legs":
        try:
            require_production_calibration(calibration, config)
        except CalibrationError as exc:
            return _failed(str(exc), calibration, debug, quality, {"image_name": image_name, "mode": mode})

    cam_enabled = bool(config.get("calibration", {}).get("camera", {}).get("enabled", False))
    work = apply_camera_undistort(image, calibration, cam_enabled)
    debug["undistorted"] = work

    if roi is None:
        roi_cfg = config.get("roi", {})
        roi_mode = str(roi_cfg.get("mode", "full")).lower()
        if roi_mode == "fixed":
            roi = (
                int(roi_cfg.get("x", 0)),
                int(roi_cfg.get("y", 0)),
                int(roi_cfg.get("width", work.shape[1])),
                int(roi_cfg.get("height", work.shape[0])),
            )
        else:
            roi = (0, 0, work.shape[1], work.shape[0])

    x, y, w, h = clip_roi(roi[0], roi[1], roi[2], roi[3], work.shape)
    roi_img, (x, y, w, h) = extract_roi(work, x, y, w, h)
    debug["roi"] = roi_img
    debug["roi_overlay"] = draw_roi(work, (x, y, w, h))

    stages = preprocess(roi_img, config)
    quality = stages.get("quality", {})
    debug["grayscale"] = stages["grayscale"]
    debug["enhanced"] = stages["enhanced"]
    debug["threshold"] = stages["threshold"]
    debug["edges"] = stages["edges"]

    warn_only = bool(config.get("quality", {}).get("warn_only", True))
    if quality.get("poor") and not warn_only:
        return _failed(
            f"Poor image quality: {quality.get('message')}",
            calibration,
            debug,
            quality,
            {"image_name": image_name, "roi": (x, y, w, h), "mode": mode},
        )

    seg = segment(stages["binary"], config)
    debug["morphology"] = seg["morphology"]
    debug["cleaned"] = seg["cleaned"]

    gray_full = to_gray(work)
    if mode == "object":
        return _measure_object(
            work,
            seg["cleaned"],
            gray_full,
            config,
            calibration,
            debug,
            quality,
            (x, y, w, h),
            image_name,
        )

    gray_roi = gray_full[y : y + h, x : x + w]
    pin_candidates = extract_pin_candidates(
        gray_roi, seg["cleaned"], config, offset=(x, y), color=roi_img
    )
    pin_pair = (
        identify_two_legs(pin_candidates, config, allow_best_pair=True)
        if len(pin_candidates) >= 2
        else PairResult(status="FAILED", message="INSPECTION FAILED — TWO LEGS NOT DETECTED", candidates=pin_candidates)
    )
    contour_candidates = extract_candidates(seg["cleaned"], config, offset=(x, y))
    contour_pair = identify_two_legs(contour_candidates, config, allow_best_pair=False)
    if (
        contour_pair.status == "OK"
        and _plausible_two_legs(contour_pair, seg["cleaned"])
        and _looks_like_separate_legs(contour_pair)
    ):
        candidates, pair = contour_candidates, contour_pair
    elif pin_pair.status == "OK":
        candidates, pair = pin_candidates, pin_pair
    elif contour_pair.status == "OK" and _plausible_two_legs(contour_pair, seg["cleaned"]):
        candidates, pair = contour_candidates, contour_pair
    else:
        candidates, pair = contour_candidates, contour_pair
    debug["pin_candidates"] = draw_candidates(work, pin_candidates)

    contour_canvas = to_bgr(work)
    for cand in candidates:
        cv2.drawContours(contour_canvas, [cand.contour.astype(np.int32)], -1, (255, 180, 0), 1)
    debug["contours"] = contour_canvas
    debug["candidates"] = draw_candidates(work, candidates)
    extra = {
        "candidate_count": len(candidates),
        "pair_score": pair.score,
        "image_name": image_name,
        "roi": (x, y, w, h),
        "status": pair.status,
        "mode": "two_legs",
    }
    if pair.status != "OK" or pair.leg1 is None or pair.leg2 is None:
        marked = draw_candidates(work, candidates)
        debug["candidates"] = marked
        debug["final"] = draw_roi(marked, (x, y, w, h))
        return _failed(pair.message, calibration, debug, quality, extra)

    leg1 = measure_leg(pair.leg1, "LEG 1", gray_full, calibration, config)
    leg2 = measure_leg(pair.leg2, "LEG 2", gray_full, calibration, config)

    c2c_px = euclidean_distance(leg1.centroid, leg2.centroid)
    try:
        edge_px = min_edge_distance(leg1.contour, leg2.contour)
    except ValueError:
        edge_px = c2c_px

    overall_w = overall_h = None
    if config.get("measurement", {}).get("overall_component", True):
        bbox = _largest_body_bbox(seg["cleaned"], (x, y), [pair.leg1, pair.leg2])
        if bbox is not None:
            overall_w, overall_h = float(bbox[2]), float(bbox[3])

    result = InspectionResult(
        status="OK",
        message="Two-leg heights and widths measured." if calibration.is_production_ready else "Two-leg heights and widths measured in pixels. Apply scale to report millimetres.",
        overall_result="OK",
        leg1=leg1,
        leg2=leg2,
        center_distance_px=c2c_px,
        center_distance_mm=_maybe_mm(c2c_px, calibration),
        edge_distance_px=edge_px,
        edge_distance_mm=_maybe_mm(edge_px, calibration),
        overall_width_px=overall_w,
        overall_height_px=overall_h,
        overall_width_mm=_maybe_mm(overall_w, calibration),
        overall_height_mm=_maybe_mm(overall_h, calibration),
        pair_score=pair.score,
        confidence=float(np.clip(pair.score, 0.0, 1.0)),
        calibration=calibration,
        quality=quality,
        debug_images=debug,
        image_name=image_name,
        roi=(x, y, w, h),
        candidate_count=len(candidates),
        mode="two_legs",
    )
    result.tolerance = evaluate_tolerances(result, config)
    result.overall_result = result.tolerance.overall if result.tolerance else "NG"
    if quality.get("poor"):
        result.message = f"Inspection complete (quality warning: {quality.get('message')})"

    overlay = draw_overlay(work, result)
    result.overlay = overlay
    result.debug_images["final"] = overlay
    logger.info(
        "Inspection %s | L1 %.4fx%.4f mm | L2 %.4fx%.4f mm | c2c %.4f mm",
        result.overall_result,
        leg1.width_mm or 0.0,
        leg1.height_mm or 0.0,
        leg2.width_mm or 0.0,
        leg2.height_mm or 0.0,
        result.center_distance_mm or 0.0,
    )
    return result
