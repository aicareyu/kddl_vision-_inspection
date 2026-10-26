"""Object measurement system — Streamlit operator UI.

Computer-vision logic lives in ``src/``. This file only handles interaction,
session state, and display.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import streamlit as st

from src import PROJECT_ROOT, configure_logging, load_config
from src.calibration import (
    CalibrationError,
    CalibrationState,
    PixelCalibration,
    calibrate_camera_from_images,
    load_state_from_config,
    save_camera_calibration,
    save_pixel_calibration,
)
from src.image_acquisition import (
    AcquisitionError,
    SUPPORTED_EXTENSIONS,
    create_demo_object,
    create_demo_two_pins,
    list_images,
    load_image,
    load_image_from_bytes,
)
from src.measurement import InspectionResult, run_inspection
from src.report import save_both, save_excel
from src.visualization import draw_overlay, draw_roi, to_rgb, zoom_pan

LOGGER = logging.getLogger("measure_app")
FILE_TYPES = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]


def _init_state() -> None:
    if "config" not in st.session_state:
        cfg = load_config()
        configure_logging(cfg)
        st.session_state.config = cfg
        st.session_state.calibration = load_state_from_config(cfg, PROJECT_ROOT)
    st.session_state.setdefault("image", None)
    st.session_state.setdefault("image_name", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("roi", None)
    st.session_state.setdefault("error", "")
    st.session_state.setdefault("notice", "")
    st.session_state.setdefault("upload_token", "")
    st.session_state.setdefault("zoom_view", 1.0)
    st.session_state.setdefault("panx_view", 0.0)
    st.session_state.setdefault("pany_view", 0.0)


def _deep_set(cfg: dict[str, Any], keys: list[str], value: Any) -> None:
    node: Any = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _badge(text: str, kind: str = "warn") -> str:
    colors = {
        "ok": ("#0f766e", "#ccfbf1"),
        "warn": ("#b45309", "#ffedd5"),
        "bad": ("#be123c", "#ffe4e6"),
        "info": ("#1d4ed8", "#dbeafe"),
    }
    bg, fg = colors.get(kind, colors["info"])
    return (
        f"<span class='pill' style='background:{bg};color:{fg}'>{text}</span>"
    )


def _fmt(value: Optional[float], unit: str, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f} {unit}"


def _apply_pixel_scale(known_mm: float, measured_px: float) -> None:
    """Save mm/px scale and re-run so overlay + metrics switch to millimetres."""
    pixel = PixelCalibration(float(known_mm), float(measured_px))
    if not pixel.is_valid:
        raise CalibrationError("Enter a known length in mm and a matching pixel length greater than zero.")
    persist = PROJECT_ROOT / st.session_state.config.get("calibration", {}).get("pixel", {}).get(
        "persist_path", "data/calibration/pixel_calibration.yaml"
    )
    save_pixel_calibration(pixel, persist)
    st.session_state.calibration = CalibrationState(
        pixel=pixel,
        camera=st.session_state.calibration.camera,
        source=str(persist),
    )
    st.session_state.error = ""
    st.session_state.notice = f"Scale applied — values now in mm ({pixel.mm_per_pixel:.6f} mm/px)"
    if st.session_state.image is not None:
        _run()


def _apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, #16324f 0%, transparent 50%),
                linear-gradient(180deg, #0b1220 0%, #111827 42%, #f3f4f6 42%);
        }
        .block-container {
            padding-top: 1.1rem;
            max-width: 1400px;
        }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #0f172a;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
            color: #94a3b8 !important;
        }

        .hero {
            background: linear-gradient(135deg, #0f766e 0%, #155e75 55%, #1e3a5f 100%);
            color: white;
            border-radius: 18px;
            padding: 22px 26px 18px;
            margin-bottom: 1.1rem;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.28);
        }
        .hero h1 {
            margin: 0;
            font-size: 1.7rem;
            letter-spacing: 0.04em;
            font-weight: 700;
        }
        .hero p { margin: 6px 0 0; opacity: 0.92; font-size: 0.98rem; }
        .hero-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }

        .pill {
            display: inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px 18px 10px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.9rem;
        }
        .card h3 {
            margin: 0 0 8px;
            font-size: 0.92rem;
            letter-spacing: 0.08em;
            color: #334155;
            text-transform: uppercase;
        }
        .result-ok, .result-ng, .result-wait {
            text-align: center;
            border-radius: 16px;
            padding: 22px 12px;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .result-ok { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .result-ng { background: #fff1f2; color: #be123c; border: 1px solid #fecdd3; }
        .result-wait { background: #f8fafc; color: #64748b; border: 1px dashed #cbd5e1; }
        div[data-testid="stMetricValue"] {
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.35rem;
        }
        .stButton > button {
            border-radius: 10px;
            height: 2.7rem;
            font-weight: 600;
        }
        .hint { color: #64748b; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _current_roi(image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    stored = st.session_state.roi
    if stored is None:
        cfg = st.session_state.config.get("roi", {})
        mode = str(cfg.get("mode", "manual")).lower()
        if mode == "fixed":
            return _clip_roi(
                int(cfg.get("x", 0)),
                int(cfg.get("y", 0)),
                int(cfg.get("width", w)),
                int(cfg.get("height", h)),
                w,
                h,
            )
        return (0, 0, w, h)
    return _clip_roi(stored[0], stored[1], stored[2], stored[3], w, h)


def _clip_roi(x: int, y: int, rw: int, rh: int, w: int, h: int) -> tuple[int, int, int, int]:
    x = int(np.clip(x, 0, max(w - 1, 0)))
    y = int(np.clip(y, 0, max(h - 1, 0)))
    rw = int(np.clip(rw, 1, max(w - x, 1)))
    rh = int(np.clip(rh, 1, max(h - y, 1)))
    return x, y, rw, rh


def _reset_view() -> None:
    st.session_state.zoom_view = 1.0
    st.session_state.panx_view = 0.0
    st.session_state.pany_view = 0.0


def _apply_demo_settings() -> None:
    cfg = st.session_state.config
    _deep_set(cfg, ["measurement", "mode"], "two_legs")
    _deep_set(cfg, ["measurement", "subpixel", "enabled"], False)
    _deep_set(cfg, ["tolerance", "leg_1", "width"], {"min": 0.04, "max": 0.50})
    _deep_set(cfg, ["tolerance", "leg_1", "height"], {"min": 0.04, "max": 0.80})
    _deep_set(cfg, ["tolerance", "leg_2", "width"], {"min": 0.04, "max": 0.50})
    _deep_set(cfg, ["tolerance", "leg_2", "height"], {"min": 0.04, "max": 0.80})
    _deep_set(cfg, ["tolerance", "distance"], {"min": 1.0, "max": 12.0})


def _load_uploaded(upload) -> None:
    try:
        image = load_image_from_bytes(upload.getvalue(), upload.name)
    except AcquisitionError as exc:
        st.session_state.error = str(exc)
        return
    st.session_state.image = image
    st.session_state.image_name = upload.name
    st.session_state.upload_token = f"{upload.name}:{len(upload.getvalue())}"
    st.session_state.result = None
    st.session_state.error = ""
    st.session_state.notice = f"Loaded {upload.name} ({image.shape[1]}×{image.shape[0]})"
    st.session_state.roi = (0, 0, image.shape[1], image.shape[0])
    _reset_view()


def _load_path(path: Path) -> None:
    try:
        image = load_image(path)
    except AcquisitionError as exc:
        st.session_state.error = str(exc)
        return
    st.session_state.image = image
    st.session_state.image_name = path.name
    st.session_state.upload_token = str(path)
    st.session_state.result = None
    st.session_state.error = ""
    st.session_state.notice = f"Loaded {path.name}"
    st.session_state.roi = (0, 0, image.shape[1], image.shape[0])
    _reset_view()


def _load_demo() -> None:
    _apply_demo_settings()
    image, truth = create_demo_two_pins()
    name = "demo_two_pins.png"
    st.session_state.calibration = CalibrationState(
        pixel=PixelCalibration(1.0, 1.0 / float(truth["mm_per_pixel"])),
        camera=st.session_state.calibration.camera,
        source="demo",
    )
    st.session_state.image = image
    st.session_state.image_name = name
    st.session_state.upload_token = name
    st.session_state.error = ""
    st.session_state.roi = (0, 0, image.shape[1], image.shape[0])
    _reset_view()
    out = PROJECT_ROOT / "data" / "input" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(str(out))
    _run()
    if st.session_state.result is not None and st.session_state.result.status == "OK":
        st.session_state.notice = "Demo ready: both pin widths, heights, and the gap are drawn on the overlay."
    else:
        st.session_state.notice = "Demo image loaded. Click Run Inspection."


def _reset() -> None:
    cfg = load_config()
    st.session_state.config = cfg
    st.session_state.calibration = load_state_from_config(cfg, PROJECT_ROOT)
    st.session_state.image = None
    st.session_state.image_name = ""
    st.session_state.upload_token = ""
    st.session_state.result = None
    st.session_state.roi = None
    st.session_state.error = ""
    st.session_state.notice = "Workspace reset."


def _run() -> None:
    image = st.session_state.image
    if image is None:
        st.session_state.error = "Upload an object image first, then click Run Inspection."
        st.session_state.notice = ""
        return
    try:
        result = run_inspection(
            image,
            st.session_state.config,
            st.session_state.calibration,
            roi=_current_roi(image),
            image_name=st.session_state.image_name,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Inspection crashed")
        st.session_state.error = f"Inspection error: {exc}"
        st.session_state.result = None
        st.session_state.notice = ""
        return
    st.session_state.result = result
    if result.status == "OK":
        st.session_state.error = ""
        st.session_state.notice = result.message
    else:
        st.session_state.error = ""
        st.session_state.notice = result.message
    if st.session_state.config.get("debug", {}).get("save_stages"):
        _dump_debug(result)


def _dump_debug(result: InspectionResult) -> None:
    stage_dir = PROJECT_ROOT / st.session_state.config.get("debug", {}).get("stage_dir", "data/output/debug")
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name, img in result.debug_images.items():
        if not isinstance(img, np.ndarray) or img.size == 0:
            continue
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        path = stage_dir / f"{name}.png"
        ok, encoded = cv2.imencode(".png", img)
        if ok:
            encoded.tofile(str(path))


def _sidebar() -> str:
    cfg = st.session_state.config
    cal: CalibrationState = st.session_state.calibration
    st.sidebar.markdown("### Control panel")
    st.sidebar.markdown(
        _badge(cal.status, "ok" if cal.is_production_ready else "warn"),
        unsafe_allow_html=True,
    )
    if cal.is_production_ready:
        st.sidebar.caption(f"Scale  {cal.mm_per_pixel:.6f} mm/px")
    else:
        st.sidebar.caption("Results in pixels until you set a tape/scale.")

    st.sidebar.divider()
    modes = ["two_legs", "object"]
    current_mode = str(cfg.get("measurement", {}).get("mode", "two_legs"))
    if current_mode not in modes:
        current_mode = "two_legs"
    mode = st.sidebar.radio(
        "Measure",
        modes,
        index=modes.index(current_mode),
        format_func=lambda v: "Two legs — each width + height + gap" if v == "two_legs" else "Single object",
    )
    _deep_set(cfg, ["measurement", "mode"], mode)

    st.sidebar.divider()
    roi_modes = ["manual", "fixed", "full"]
    current_roi = str(cfg.get("roi", {}).get("mode", "manual"))
    if current_roi not in roi_modes:
        current_roi = "manual"
    roi_mode = st.sidebar.radio("ROI", roi_modes, index=roi_modes.index(current_roi))
    _deep_set(cfg, ["roi", "mode"], roi_mode)

    image = st.session_state.image
    if image is not None and roi_mode == "manual":
        h, w = image.shape[:2]
        current = _current_roi(image)
        token = st.session_state.image_name or "img"
        x = st.sidebar.slider("ROI X", 0, max(w - 1, 0), int(current[0]), key=f"roi_x_{token}_{w}")
        y = st.sidebar.slider("ROI Y", 0, max(h - 1, 0), int(current[1]), key=f"roi_y_{token}_{h}")
        rw = st.sidebar.slider("ROI width", 1, w, int(min(current[2], w)), key=f"roi_w_{token}_{w}")
        rh = st.sidebar.slider("ROI height", 1, h, int(min(current[3], h)), key=f"roi_h_{token}_{h}")
        st.session_state.roi = _clip_roi(x, y, rw, rh, w, h)
    elif image is not None and roi_mode == "fixed":
        r = cfg.get("roi", {})
        st.session_state.roi = _clip_roi(
            int(r.get("x", 0)),
            int(r.get("y", 0)),
            int(r.get("width", image.shape[1])),
            int(r.get("height", image.shape[0])),
            image.shape[1],
            image.shape[0],
        )
        st.sidebar.caption(f"Config ROI {st.session_state.roi}")
    elif image is not None:
        st.session_state.roi = (0, 0, image.shape[1], image.shape[0])

    with st.sidebar.expander("Image processing", expanded=False):
        methods = ["adaptive", "otsu", "global", "canny"]
        method = str(cfg.get("segmentation", {}).get("method", "adaptive"))
        if method not in methods:
            method = "adaptive"
        method = st.selectbox("Segmentation", methods, index=methods.index(method))
        _deep_set(cfg, ["segmentation", "method"], method)
        invert = st.checkbox("Invert binary (dark object on light)", value=bool(cfg.get("segmentation", {}).get("invert_binary", True)))
        _deep_set(cfg, ["segmentation", "invert_binary"], invert)
        _deep_set(cfg, ["preprocessing", "blur", "enabled"], st.checkbox("Blur", value=bool(cfg.get("preprocessing", {}).get("blur", {}).get("enabled", True))))
        _deep_set(cfg, ["preprocessing", "clahe", "enabled"], st.checkbox("CLAHE", value=bool(cfg.get("preprocessing", {}).get("clahe", {}).get("enabled", True))))
        _deep_set(cfg, ["segmentation", "morphology", "enabled"], st.checkbox("Morphology", value=bool(cfg.get("segmentation", {}).get("morphology", {}).get("enabled", True))))
        _deep_set(cfg, ["measurement", "subpixel", "enabled"], st.checkbox("Sub-pixel", value=bool(cfg.get("measurement", {}).get("subpixel", {}).get("enabled", True))))
        _deep_set(cfg, ["debug", "enabled"], st.checkbox("Debug stages", value=bool(cfg.get("debug", {}).get("enabled", False))))

    if mode == "object":
        with st.sidebar.expander("Object filters / tolerances", expanded=False):
            obj = cfg.setdefault("object_detection", {})
            obj["min_area"] = st.number_input("min_area", 1.0, 5e7, float(obj.get("min_area", 200)), 10.0)
            obj["max_area"] = st.number_input("max_area", 1.0, 5e7, float(obj.get("max_area", 5e7)), 100.0)
            tol = cfg.setdefault("tolerance", {}).setdefault("object", {})
            tol["enabled"] = st.checkbox("Enable OK/NG limits (mm)", value=bool(tol.get("enabled", False)))
            w = tol.setdefault("width", {})
            h = tol.setdefault("height", {})
            cols = st.columns(2)
            w["min"] = cols[0].number_input("W min", value=float(w.get("min", 0.0)), format="%.3f", key="obj_wmin")
            w["max"] = cols[1].number_input("W max", value=float(w.get("max", 10000.0)), format="%.3f", key="obj_wmax")
            cols = st.columns(2)
            h["min"] = cols[0].number_input("H min", value=float(h.get("min", 0.0)), format="%.3f", key="obj_hmin")
            h["max"] = cols[1].number_input("H max", value=float(h.get("max", 10000.0)), format="%.3f", key="obj_hmax")
    else:
        with st.sidebar.expander("Leg filters / tolerances", expanded=False):
            det = cfg.setdefault("leg_detection", {})
            det["min_area"] = st.number_input("min_area", 1.0, 1_000_000.0, float(det.get("min_area", 30)), 1.0)
            det["max_area"] = st.number_input("max_area", 1.0, 1_000_000.0, float(det.get("max_area", 8000)), 10.0)
            det["min_width"] = st.number_input("min_width", 0.0, 2000.0, float(det.get("min_width", 2)), 0.5)
            det["max_width"] = st.number_input("max_width", 0.0, 2000.0, float(det.get("max_width", 120)), 1.0)
            det["min_height"] = st.number_input("min_height", 0.0, 4000.0, float(det.get("min_height", 4)), 0.5)
            det["max_height"] = st.number_input("max_height", 0.0, 4000.0, float(det.get("max_height", 250)), 1.0)
            tol = cfg.setdefault("tolerance", {})
            for leg_key, title in (("leg_1", "Leg 1"), ("leg_2", "Leg 2")):
                st.caption(title)
                block = tol.setdefault(leg_key, {})
                w = block.setdefault("width", {})
                h = block.setdefault("height", {})
                cols = st.columns(2)
                w["min"] = cols[0].number_input(f"{title} W min", value=float(w.get("min", 0.10)), format="%.3f", key=f"{leg_key}_wmin")
                w["max"] = cols[1].number_input(f"{title} W max", value=float(w.get("max", 0.15)), format="%.3f", key=f"{leg_key}_wmax")
                cols = st.columns(2)
                h["min"] = cols[0].number_input(f"{title} H min", value=float(h.get("min", 0.20)), format="%.3f", key=f"{leg_key}_hmin")
                h["max"] = cols[1].number_input(f"{title} H max", value=float(h.get("max", 0.30)), format="%.3f", key=f"{leg_key}_hmax")
            dist = tol.setdefault("distance", {})
            cols = st.columns(2)
            dist["min"] = cols[0].number_input("Distance min", value=float(dist.get("min", 0.50)), format="%.3f", key="dmin")
            dist["max"] = cols[1].number_input("Distance max", value=float(dist.get("max", 0.60)), format="%.3f", key="dmax")
    return mode


def _reference_px(result: Optional[InspectionResult]) -> tuple[float, str]:
    """Pixel length used to convert the current photo to millimetres."""
    if result is None:
        return 0.0, "none"
    if result.center_distance_px and result.center_distance_px > 1:
        return float(result.center_distance_px), "inter-leg D"
    if result.overall_width_px and result.overall_width_px > 1:
        return float(result.overall_width_px), "plate length"
    if result.object is not None and result.object.width_px > 1:
        return float(result.object.width_px), "object width"
    return 0.0, "none"


def _calibration_card() -> None:
    cal: CalibrationState = st.session_state.calibration
    result: Optional[InspectionResult] = st.session_state.result
    st.markdown("<div class='card'><h3>Scale — convert pixels to mm</h3>", unsafe_allow_html=True)
    st.caption("Type the real length of something already measured in this photo (D or the plate).")
    ref_px, ref_name = _reference_px(result)
    default_known = cal.pixel.known_distance_mm if cal.pixel and cal.pixel.known_distance_mm > 0 else 10.0
    default_px = ref_px if ref_px > 0 else (cal.pixel.measured_distance_px if cal.pixel else 0.0)
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    known = c1.number_input("Known length (mm)", min_value=0.0, value=float(default_known), format="%.3f", key="scale_known_mm")
    measured = c2.number_input(
        f"Same length (px){' — ' + ref_name if ref_name != 'none' else ''}",
        min_value=0.0,
        value=float(default_px),
        format="%.2f",
        key="scale_known_px",
    )
    if c3.button("Show in mm", type="primary", width="stretch"):
        try:
            _apply_pixel_scale(known, measured)
            st.rerun()
        except CalibrationError as exc:
            st.session_state.error = str(exc)

    if st.session_state.calibration.is_production_ready:
        factor = st.session_state.calibration.mm_per_pixel
        st.success(f"Scale ready: 100 px = {100.0 * factor:.3f} mm")

    with st.expander("Advanced: printed checkerboard (lens only — skip for object photos)"):
        st.caption("Do not upload the object or phone photo here.")
        cam_cfg = st.session_state.config.get("calibration", {}).get("camera", {})
        cols = st.columns(4)
        board_c = cols[0].number_input("Corners X", 2, 20, int(cam_cfg.get("checkerboard_cols", 9)))
        board_r = cols[1].number_input("Corners Y", 2, 20, int(cam_cfg.get("checkerboard_rows", 6)))
        square = cols[2].number_input("Square mm", 0.01, 50.0, float(cam_cfg.get("square_size_mm", 1.0)), format="%.3f")
        enable_undistort = cols[3].checkbox("Undistort", value=bool(cam_cfg.get("enabled", False)))
        _deep_set(st.session_state.config, ["calibration", "camera", "enabled"], enable_undistort)
        uploads = st.file_uploader("Checkerboard images (3+)", type=FILE_TYPES, accept_multiple_files=True)
        if st.button("Run camera calibration"):
            if not uploads or len(uploads) < 3:
                st.warning("Not needed for object measurement. This needs 3+ printed chessboard photos.")
            else:
                try:
                    frames = [load_image_from_bytes(u.getvalue(), u.name) for u in uploads]
                    cam = calibrate_camera_from_images(frames, int(board_c), int(board_r), float(square))
                    persist = PROJECT_ROOT / cam_cfg.get("persist_path", "data/calibration/camera_calibration.yaml")
                    save_camera_calibration(cam, persist)
                    st.session_state.calibration.camera = cam
                    st.success(f"Lens model saved. RMS = {cam.rms_error:.4f} px")
                except (CalibrationError, AcquisitionError) as exc:
                    st.error(str(exc))
    st.markdown("</div>", unsafe_allow_html=True)


def _show_image_pair() -> None:
    image = st.session_state.image
    result: Optional[InspectionResult] = st.session_state.result
    st.markdown("<div class='card'><h3>Images</h3>", unsafe_allow_html=True)
    if image is None:
        st.markdown(
            "<div class='result-wait'>Upload an object photo to begin</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    z1, z2, z3, z4 = st.columns([1, 1, 1, 1.4])
    zoom = z1.slider("Zoom", 1.0, 4.0, 1.0, 0.1, key="zoom_view")
    pan_x = z2.slider("Pan X", -1.0, 1.0, 0.0, 0.05, key="panx_view")
    pan_y = z3.slider("Pan Y", -1.0, 1.0, 0.0, 0.05, key="pany_view")
    z4.caption(f"{st.session_state.image_name}  ·  {image.shape[1]}×{image.shape[0]} px")

    left, right = st.columns(2)
    roi = _current_roi(image)
    original = draw_roi(image, roi)
    with left:
        st.markdown("**Original + ROI**")
        st.image(to_rgb(zoom_pan(original, zoom, pan_x, pan_y)), width="stretch")
    with right:
        st.markdown("**Result overlay**")
        if result is not None and (result.leg1 is not None or result.object is not None):
            live = draw_overlay(image, result)
            result.overlay = live
            st.image(to_rgb(zoom_pan(live, zoom, pan_x, pan_y)), width="stretch")
            st.caption("Red = pin height (H). Blue = pin width (W) and Inter-Leg Distance (D).")
        elif result is not None and result.overlay is not None:
            st.image(to_rgb(zoom_pan(result.overlay, zoom, pan_x, pan_y)), width="stretch")
            st.caption("Red = pin height (H). Blue = pin width (W) and Inter-Leg Distance (D).")
        else:
            st.image(to_rgb(zoom_pan(original, zoom, pan_x, pan_y)), width="stretch")
            st.caption("Overlay with bounding boxes appears after Run Inspection.")
    st.markdown("</div>", unsafe_allow_html=True)


def _result_box(label: str) -> None:
    cls = "result-ok" if label == "OK" else "result-ng" if label in {"NG", "FAILED", "REVIEW"} else "result-wait"
    st.markdown(f"<div class='{cls}'>{label}</div>", unsafe_allow_html=True)


def _show_measurements(result: InspectionResult) -> None:
    st.markdown("<div class='card'><h3>Measurements</h3>", unsafe_allow_html=True)
    factor = result.calibration.mm_per_pixel if result.calibration.is_production_ready else None
    if factor:
        st.caption(f"{result.calibration.status}  ·  {factor:.6f} mm/px")
    else:
        st.caption("Values are in pixels. Enter the real D or plate length (mm) in Scale above, then click Show in mm.")
        ref_px, ref_name = _reference_px(result)
        if ref_px > 0:
            cols = st.columns([1.4, 1, 1])
            known_mm = cols[0].number_input(
                f"Real {ref_name} (mm)",
                min_value=0.0,
                value=10.0,
                format="%.3f",
                key="measure_known_mm",
            )
            cols[1].metric("Measured", _fmt(ref_px, "px", 2))
            if cols[2].button("Convert to mm", type="primary", width="stretch", key="convert_mm_now"):
                try:
                    _apply_pixel_scale(known_mm, ref_px)
                    st.rerun()
                except CalibrationError as exc:
                    st.session_state.error = str(exc)

    if result.mode == "object":
        if result.object is None:
            st.warning(result.message)
            _result_box(result.status)
            st.markdown("</div>", unsafe_allow_html=True)
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Width", _fmt(result.object.width_mm, "mm"), _fmt(result.object.width_px, "px", 1))
        c2.metric("Height", _fmt(result.object.height_mm, "mm"), _fmt(result.object.height_px, "px", 1))
        c3.metric("Angle", f"{result.object.angle_deg:.2f}°")
        if result.object_area_mm2 is not None:
            c4.metric("Area", _fmt(result.object_area_mm2, "mm²"))
        else:
            c4.metric("Area", _fmt(result.object_area_px, "px²", 0))
        badge = result.overall_result if result.status == "OK" else result.status
        _result_box(badge)
        st.caption(result.message)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if result.leg1 is None or result.leg2 is None:
        st.warning(result.message)
        _result_box(result.status)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Leg 1 (Front)**")
        st.metric("HEIGHT", _fmt(result.leg1.height_mm, "mm"), _fmt(result.leg1.height_px, "px", 2))
        st.metric("Width", _fmt(result.leg1.width_mm, "mm"), _fmt(result.leg1.width_px, "px", 2))
    with c2:
        st.markdown("**Leg 2 (Back)**")
        st.metric("HEIGHT", _fmt(result.leg2.height_mm, "mm"), _fmt(result.leg2.height_px, "px", 2))
        st.metric("Width", _fmt(result.leg2.width_mm, "mm"), _fmt(result.leg2.width_px, "px", 2))
    with c3:
        st.markdown("**Inter-leg distance (D)**")
        st.metric("D  (center-to-center)", _fmt(result.center_distance_mm, "mm"), _fmt(result.center_distance_px, "px", 2))
        st.metric("Edge-to-edge", _fmt(result.edge_distance_mm, "mm"), _fmt(result.edge_distance_px, "px", 2))
    if result.tolerance:
        t1, t2, t3 = st.columns(3)
        t1.markdown(f"Leg 1 {_badge(result.tolerance.leg1, 'ok' if result.tolerance.leg1 == 'OK' else 'bad')}", unsafe_allow_html=True)
        t2.markdown(f"Leg 2 {_badge(result.tolerance.leg2, 'ok' if result.tolerance.leg2 == 'OK' else 'bad')}", unsafe_allow_html=True)
        t3.markdown(f"Distance {_badge(result.tolerance.distance, 'ok' if result.tolerance.distance == 'OK' else 'bad')}", unsafe_allow_html=True)
    badge = result.overall_result if result.status == "OK" else result.status
    if result.status == "REVIEW":
        badge = "REVIEW"
    _result_box(badge)
    st.caption(result.message)
    st.markdown("</div>", unsafe_allow_html=True)


def _show_debug(result: InspectionResult) -> None:
    if not st.session_state.config.get("debug", {}).get("enabled"):
        return
    st.markdown("<div class='card'><h3>Debug stages</h3>", unsafe_allow_html=True)
    order = [
        ("original", "1. Original"),
        ("roi", "2. ROI"),
        ("grayscale", "3. Grayscale"),
        ("enhanced", "4. Enhanced"),
        ("threshold", "5. Threshold"),
        ("morphology", "6. Morphology"),
        ("edges", "7. Edges"),
        ("contours", "8. Contours"),
        ("candidates", "9. Candidates"),
        ("pin_candidates", "9b. Pin candidates"),
        ("final", "10. Final"),
    ]
    cols = st.columns(2)
    for i, (key, title) in enumerate(order):
        img = result.debug_images.get(key)
        if img is None:
            continue
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            st.image(to_rgb(img), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Two-Leg Measurement", layout="wide", page_icon="▣")
    _apply_custom_css()
    _init_state()
    _sidebar()

    cal: CalibrationState = st.session_state.calibration
    cal_kind = "ok" if cal.is_production_ready else "warn"
    st.markdown(
        f"""
        <div class="hero">
            <h1>TWO-LEG MEASUREMENT</h1>
            <p>Finds both pins/legs and reports each WIDTH, each height, and the gap between them. No size input needed.</p>
            <div class="hero-row">
                {_badge("1  Load demo or upload", "info")}
                {_badge("2  Boxes + WIDTH lines", "info")}
                {_badge(cal.status, cal_kind)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'><h3>Image input</h3>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns([2.1, 2.1, 1.1])
    with i1:
        upload = st.file_uploader("Upload backside image (two legs)", type=FILE_TYPES)
        if upload is not None:
            token = f"{upload.name}:{len(upload.getvalue())}"
            if st.session_state.upload_token != token:
                _load_uploaded(upload)
    with i2:
        folder = st.text_input("Folder", value=str(PROJECT_ROOT / "data" / "input"))
        files = list_images(folder)
        names = ["—"] + [p.name for p in files]
        choice = st.selectbox("Choose from folder", names)
        if choice != "—" and st.button("Load selected", width="stretch"):
            match = next(p for p in files if p.name == choice)
            _load_path(match)
    with i3:
        st.write("")
        if st.button("Load demo", type="primary", width="stretch"):
            _load_demo()
            st.rerun()
        st.caption("Loads a brass plate, finds both pins, and draws WIDTH boxes.")
    st.markdown("</div>", unsafe_allow_html=True)

    _calibration_card()

    st.markdown("<div class='card'><h3>Actions</h3>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Run Inspection", type="primary", width="stretch"):
        _run()
    if b2.button("Reset", width="stretch"):
        _reset()
        st.rerun()
    result: Optional[InspectionResult] = st.session_state.result
    if b3.button("Save CSV", width="stretch", disabled=result is None):
        if result is not None:
            try:
                csv_path, xls_path = save_both(result, st.session_state.config, PROJECT_ROOT, st.session_state.image_name)
                if xls_path is None:
                    st.session_state.notice = f"Saved {csv_path.name}"
                else:
                    st.session_state.notice = f"Saved {csv_path.name} and {xls_path.name}"
            except Exception as exc:  # noqa: BLE001
                st.session_state.error = f"Save failed: {exc}"
    if b4.button("Export Excel", width="stretch", disabled=result is None):
        if result is not None:
            try:
                xls = PROJECT_ROOT / st.session_state.config.get("reporting", {}).get("excel_path", "data/output/inspections.xlsx")
                save_excel(result, xls, st.session_state.image_name)
                st.session_state.notice = f"Exported {xls.name}"
            except Exception as exc:  # noqa: BLE001
                st.session_state.error = f"Excel export failed: {exc}"
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.notice:
        st.info(st.session_state.notice)

    _show_image_pair()
    if result is not None:
        _show_measurements(result)
        _show_debug(result)


if __name__ == "__main__":
    main()
