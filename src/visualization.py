"""Measurement overlay and debug-stage rendering."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .measurement import InspectionResult, LegMeasurement


LEG1_COLOR = (40, 180, 70)
LEG2_COLOR = (40, 140, 230)
WIDTH_COLOR = (255, 0, 0)  # blue (BGR) — sample width / inter-leg
HEIGHT_COLOR = (0, 0, 255)  # red (BGR) — sample pin height
DIST_COLOR = (255, 0, 0)
ROI_COLOR = (0, 200, 220)
OK_COLOR = (40, 180, 70)
NG_COLOR = (40, 40, 220)
TEXT_BG = (20, 20, 20)
INK = (25, 25, 25)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _xy(point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 1,
    *,
    bg: bool = True,
) -> None:
    """Readable label: dark filled box + white text, clipped to the image."""
    h, w = canvas.shape[:2]
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    x = int(np.clip(x, pad, max(pad, w - tw - pad)))
    y = int(np.clip(y, th + pad, max(th + pad, h - baseline - pad)))
    if bg:
        x1, y1 = max(0, x - pad), max(0, y - th - pad)
        x2, y2 = min(w - 1, x + tw + pad), min(h - 1, y + baseline + pad)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), TEXT_BG, -1)
        cv2.putText(canvas, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    else:
        cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    canvas = to_bgr(image)
    x, y, w, h = roi
    cv2.rectangle(canvas, (x, y), (x + w, y + h), ROI_COLOR, 2)
    _put_text(canvas, "ROI", (x + 8, y + 22), ROI_COLOR)
    return canvas


def _fmt_mm(value: Optional[float], px: Optional[float]) -> str:
    if value is not None:
        return f"{value:.3f} mm"
    if px is not None:
        return f"{px:.2f} px"
    return "n/a"


def _halo_line(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    cv2.line(canvas, p1, p2, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


def _halo_rect(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), thickness + 3)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)


def _draw_rotated_box(canvas: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return
    cv2.polylines(canvas, [pts], True, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)


def _padded_bbox(contour: np.ndarray, pad: int = 10) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    x = max(0, x - pad)
    y = max(0, y - pad)
    return x, y, w + 2 * pad, h + 2 * pad


def _dimension_line(
    canvas: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    offset: float,
) -> None:
    """Engineering-style dimension: extension ticks, span line, and value."""
    start = np.asarray(a, dtype=np.float64).reshape(2)
    end = np.asarray(b, dtype=np.float64).reshape(2)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1.0:
        return
    direction = vec / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    shifted_a = start + normal * offset
    shifted_b = end + normal * offset
    pa, pb = _xy(shifted_a), _xy(shifted_b)
    _halo_line(canvas, _xy(start), pa, color, 2)
    _halo_line(canvas, _xy(end), pb, color, 2)
    _halo_line(canvas, pa, pb, color, 3)
    tick = normal * 7.0
    _halo_line(canvas, _xy(shifted_a - tick), _xy(shifted_a + tick), color, 2)
    _halo_line(canvas, _xy(shifted_b - tick), _xy(shifted_b + tick), color, 2)
    mid = (shifted_a + shifted_b) / 2.0
    _put_text(canvas, label, (int(mid[0]) - 20, int(mid[1]) - 6), color, 0.48)


def _ink_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = INK,
    scale: float = 0.48,
) -> None:
    """Sample-style label: white halo, dark ink, no filled box."""
    x, y = origin
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x = int(np.clip(x, 4, max(4, w - tw - 4)))
    y = int(np.clip(y, th + 4, max(th + 4, h - 6)))
    cv2.putText(canvas, text, (x, y), font, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def _pin_extents(leg: LegMeasurement) -> dict[str, np.ndarray]:
    pts = np.asarray(leg.contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        c = np.array(leg.centroid, dtype=np.float64)
        return {"tip": c, "base": c, "left": c, "right": c, "center": c}
    return {
        "tip": pts[int(np.argmin(pts[:, 1]))],
        "base": pts[int(np.argmax(pts[:, 1]))],
        "left": pts[int(np.argmin(pts[:, 0]))],
        "right": pts[int(np.argmax(pts[:, 0]))],
        "center": np.array(leg.centroid, dtype=np.float64),
    }


def _draw_end_ticks(canvas: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], size: float = 6.0) -> None:
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1:
        return
    normal = np.array([-vec[1], vec[0]], dtype=np.float64) / length * size
    for point in (a, b):
        cv2.line(canvas, _xy(point - normal), _xy(point + normal), color, 2, cv2.LINE_AA)


def _annotate_leg_sample(canvas: np.ndarray, leg: LegMeasurement, *, front: bool) -> None:
    """Red height I-beam (tip to base) and blue width caliper on the pin."""
    center = np.array(leg.centroid, dtype=np.float64)
    scale = max(0.85, min(canvas.shape[:2]) / 480.0)
    if leg.tip is not None and leg.base is not None:
        tip = np.array(leg.tip, dtype=np.float64)
        base = np.array(leg.base, dtype=np.float64)
    else:
        half = max(6.0, float(leg.height_px) * 0.5)
        tip = np.array([center[0], center[1] - half], dtype=np.float64)
        base = np.array([center[0], center[1] + half], dtype=np.float64)
    cap = 5.0 * scale
    jaw = 5.0 * scale
    cal_w = max(8.0 * scale, float(leg.width_px))
    side = -1.0 if front else 1.0
    beam_gap = max(10.0 * scale, 0.6 * cal_w + 8.0 * scale)
    beam_x = center[0] + side * beam_gap
    y_top, y_bot = float(min(tip[1], base[1])), float(max(tip[1], base[1]))
    cv2.line(canvas, _xy((beam_x, y_top)), _xy((beam_x, y_bot)), HEIGHT_COLOR, 2, cv2.LINE_AA)
    cv2.line(canvas, _xy((beam_x - cap, y_top)), _xy((beam_x + cap, y_top)), HEIGHT_COLOR, 2, cv2.LINE_AA)
    cv2.line(canvas, _xy((beam_x - cap, y_bot)), _xy((beam_x + cap, y_bot)), HEIGHT_COLOR, 2, cv2.LINE_AA)

    y_w = float(max(tip[1], base[1])) + 6.0 * scale
    x0, x1, xm = center[0] - cal_w / 2.0, center[0] + cal_w / 2.0, center[0]
    cv2.line(canvas, _xy((x0, y_w)), _xy((x1, y_w)), WIDTH_COLOR, 2, cv2.LINE_AA)
    for xx in (x0, xm, x1):
        cv2.line(canvas, _xy((xx, y_w - jaw)), _xy((xx, y_w + jaw)), WIDTH_COLOR, 2, cv2.LINE_AA)

    name = "Leg 1 (Front)" if front else "Leg 2 (Back)"
    h_txt = f"H {_fmt_mm(leg.height_mm, leg.height_px)}"
    w_txt = f"W {_fmt_mm(leg.width_mm, leg.width_px)}"
    text_scale = 0.46 * scale
    if front:
        _ink_text(canvas, name, (int(center[0] - 78 * scale), int(y_bot + 20 * scale)), INK, text_scale)
        _ink_text(canvas, h_txt, (int(beam_x - 52 * scale), int((y_top + y_bot) / 2.0)), HEIGHT_COLOR, text_scale)
        _ink_text(canvas, w_txt, (int(center[0] - 20 * scale), int(y_w + 16 * scale)), WIDTH_COLOR, text_scale)
    else:
        _ink_text(canvas, name, (int(center[0] + 8 * scale), int(y_top - 8 * scale)), INK, text_scale)
        _ink_text(canvas, h_txt, (int(beam_x + 8 * scale), int((y_top + y_bot) / 2.0)), HEIGHT_COLOR, text_scale)
        _ink_text(canvas, w_txt, (int(center[0] - 10 * scale), int(y_w + 16 * scale)), WIDTH_COLOR, text_scale)


def _annotate_leg(canvas: np.ndarray, leg: LegMeasurement, color: tuple[int, int, int], stack: int) -> None:
    if leg.contour is None or len(leg.contour) < 3:
        return
    box = _padded_bbox(leg.contour, pad=10)
    _halo_rect(canvas, box, color, 3)
    cv2.drawContours(canvas, [leg.contour.astype(np.int32)], -1, color, 2, cv2.LINE_AA)
    cx, cy = int(leg.centroid[0]), int(leg.centroid[1])
    cv2.circle(canvas, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
    bx, by, bw, _bh = box
    tag_x = bx if stack == 0 else max(8, bx + bw - 64)
    tag_y = max(22, by - 8)
    _put_text(canvas, leg.label, (tag_x, tag_y), color, 0.55, 2)
    _put_text(canvas, f"W {_fmt_mm(leg.width_mm, leg.width_px)}", (tag_x, tag_y + 20), WIDTH_COLOR, 0.45)


def draw_overlay(image: np.ndarray, result: InspectionResult) -> np.ndarray:
    canvas = to_bgr(image)
    if result.mode == "two_legs" and result.leg1 is not None and result.leg2 is not None:
        return _draw_two_leg_sample(canvas, result)

    if result.roi is not None:
        x, y, w, h = result.roi
        cv2.rectangle(canvas, (x, y), (x + w, y + h), ROI_COLOR, 2)

    if result.object is not None:
        _annotate_leg(canvas, result.object, LEG1_COLOR, 0)
    _draw_hud(canvas, result)
    badge = result.overall_result if result.status == "OK" else result.status
    badge_color = OK_COLOR if badge == "OK" else NG_COLOR
    cv2.rectangle(canvas, (12, 12), (250, 62), badge_color, -1)
    cv2.putText(canvas, f"RESULT: {badge}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _draw_two_leg_sample(canvas: np.ndarray, result: InspectionResult) -> np.ndarray:
    """Match the sample: I-beams, calipers, tip-to-tip D with a midpoint dot."""
    assert result.leg1 is not None and result.leg2 is not None
    _annotate_leg_sample(canvas, result.leg1, front=True)
    _annotate_leg_sample(canvas, result.leg2, front=False)

    tip1 = np.array(result.leg1.centroid, dtype=np.float64)
    tip2 = np.array(result.leg2.centroid, dtype=np.float64)
    cv2.line(canvas, _xy(tip1), _xy(tip2), DIST_COLOR, 2, cv2.LINE_AA)
    mid = (tip1 + tip2) / 2.0
    cv2.circle(canvas, _xy(mid), 5, DIST_COLOR, -1, cv2.LINE_AA)
    gap = _fmt_mm(result.center_distance_mm, result.center_distance_px)
    _ink_text(canvas, "Inter-Leg Distance (D)", (int(mid[0]) - 120, int(mid[1]) - 18), DIST_COLOR, 0.50)
    _ink_text(canvas, gap, (int(mid[0]) - 40, int(mid[1]) + 20), DIST_COLOR, 0.48)
    return canvas


def _draw_hud(canvas: np.ndarray, result: InspectionResult) -> None:
    """Fixed operator panel — values stay readable even when legs occupy few pixels."""
    lines = ["MEASUREMENT HUD"]
    if result.object is not None:
        lines.append(f"OBJECT  W {_fmt_mm(result.object.width_mm, result.object.width_px)}")
        lines.append(f"        H {_fmt_mm(result.object.height_mm, result.object.height_px)}")
        if result.object_area_px is not None:
            area = f"{result.object_area_mm2:.3f} mm2" if result.object_area_mm2 is not None else f"{result.object_area_px:.1f} px2"
            lines.append(f"AREA   {area}")
    if result.leg1 is not None:
        lines.append(f"LEG 1  W {_fmt_mm(result.leg1.width_mm, result.leg1.width_px)}  H {_fmt_mm(result.leg1.height_mm, result.leg1.height_px)}")
    if result.leg2 is not None:
        lines.append(f"LEG 2  W {_fmt_mm(result.leg2.width_mm, result.leg2.width_px)}  H {_fmt_mm(result.leg2.height_mm, result.leg2.height_px)}")
    if result.center_distance_px is not None:
        lines.append(f"GAP C2C {_fmt_mm(result.center_distance_mm, result.center_distance_px)}")
        lines.append(f"GAP EDGE {_fmt_mm(result.edge_distance_mm, result.edge_distance_px)}")
    panel_w = min(360, max(160, canvas.shape[1] - 24))
    line_h = 22
    panel_h = 16 + line_h * len(lines)
    x0 = max(8, canvas.shape[1] - panel_w - 12)
    y0 = 12
    if y0 + panel_h >= canvas.shape[0]:
        y0 = max(8, canvas.shape[0] - panel_h - 8)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)
    for i, line in enumerate(lines):
        color = (255, 255, 255) if i == 0 else (230, 230, 230)
        cv2.putText(canvas, line, (x0 + 10, y0 + 20 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def zoom_pan(image: np.ndarray, zoom: float, pan_x: float, pan_y: float) -> np.ndarray:
    """Crop a zoomed viewport. ``pan_*`` are fractions of leftover margin in [-1, 1]."""
    if image is None or image.size == 0:
        return image
    zoom = max(1.0, float(zoom))
    h, w = image.shape[:2]
    vw, vh = max(1, int(w / zoom)), max(1, int(h / zoom))
    max_x = max(0, w - vw)
    max_y = max(0, h - vh)
    x = int((pan_x + 1.0) / 2.0 * max_x)
    y = int((pan_y + 1.0) / 2.0 * max_y)
    x = int(np.clip(x, 0, max_x))
    y = int(np.clip(y, 0, max_y))
    return image[y : y + vh, x : x + vw].copy()
