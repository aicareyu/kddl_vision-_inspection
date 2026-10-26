"""Pixel-to-mm scale and OpenCV camera calibration.

A valid pixel scale is mandatory for production measurement. Camera-matrix
calibration (checkerboard) is optional and used for undistortion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Invalid or missing calibration."""


@dataclass
class PixelCalibration:
    """Simple scale: ``mm_per_pixel = known_mm / measured_px``."""

    known_distance_mm: float
    measured_distance_px: float
    method: str = "reference_length"

    @property
    def mm_per_pixel(self) -> float:
        if self.measured_distance_px <= 0:
            raise CalibrationError("Measured reference distance in pixels must be > 0.")
        if self.known_distance_mm <= 0:
            raise CalibrationError("Known reference distance in mm must be > 0.")
        return float(self.known_distance_mm / self.measured_distance_px)

    @property
    def is_valid(self) -> bool:
        try:
            factor = self.mm_per_pixel
        except CalibrationError:
            return False
        return np.isfinite(factor) and factor > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_distance_mm": float(self.known_distance_mm),
            "measured_distance_px": float(self.measured_distance_px),
            "mm_per_pixel": float(self.mm_per_pixel) if self.is_valid else None,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PixelCalibration":
        return cls(
            known_distance_mm=float(data.get("known_distance_mm", 0.0)),
            measured_distance_px=float(data.get("measured_distance_px", 0.0)),
            method=str(data.get("method", "reference_length")),
        )


@dataclass
class CameraCalibration:
    """OpenCV pinhole + distortion model."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms_error: float
    checkerboard: tuple[int, int] = (9, 6)
    square_size_mm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "rms_error": float(self.rms_error),
            "checkerboard": [int(self.checkerboard[0]), int(self.checkerboard[1])],
            "square_size_mm": float(self.square_size_mm),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            rms_error=float(data.get("rms_error", 0.0)),
            checkerboard=tuple(data.get("checkerboard", [9, 6])),  # type: ignore[arg-type]
            square_size_mm=float(data.get("square_size_mm", 1.0)),
        )


@dataclass
class CalibrationState:
    """Combined calibration used by the inspection engine."""

    pixel: Optional[PixelCalibration] = None
    camera: Optional[CameraCalibration] = None
    source: str = "none"

    @property
    def status(self) -> str:
        return "CALIBRATED" if self.is_production_ready else "NOT CALIBRATED"

    @property
    def is_production_ready(self) -> bool:
        return self.pixel is not None and self.pixel.is_valid

    @property
    def mm_per_pixel(self) -> float:
        if self.pixel is None or not self.pixel.is_valid:
            raise CalibrationError("Pixel-to-mm calibration is missing or invalid.")
        return self.pixel.mm_per_pixel

    def pixels_to_mm(self, pixels: float) -> float:
        return float(pixels) * self.mm_per_pixel

    def mm_to_pixels(self, millimetres: float) -> float:
        return float(millimetres) / self.mm_per_pixel


def compute_mm_per_pixel(known_distance_mm: float, measured_distance_px: float) -> float:
    """Pure helper used by tests and the UI."""
    return PixelCalibration(known_distance_mm, measured_distance_px).mm_per_pixel


def load_pixel_calibration(path: str | Path) -> Optional[PixelCalibration]:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    calib = PixelCalibration.from_dict(data)
    return calib if calib.is_valid else None


def save_pixel_calibration(calib: PixelCalibration, path: str | Path) -> Path:
    if not calib.is_valid:
        raise CalibrationError("Refusing to save an invalid pixel calibration.")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = calib.to_dict()
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    logger.info("Saved pixel calibration to %s (%.6f mm/px)", file_path, calib.mm_per_pixel)
    return file_path


def load_camera_calibration(path: str | Path) -> Optional[CameraCalibration]:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    try:
        return CameraCalibration.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Camera calibration file is invalid: %s", exc)
        return None


def save_camera_calibration(calib: CameraCalibration, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(calib.to_dict(), handle, sort_keys=False)
    logger.info("Saved camera calibration to %s (RMS=%.4f)", file_path, calib.rms_error)
    return file_path


def load_state_from_config(config: dict[str, Any], project_root: Path) -> CalibrationState:
    """Restore persisted calibration described by ``config.yaml``."""
    cal_cfg = config.get("calibration", {})
    pixel_cfg = cal_cfg.get("pixel", {})
    cam_cfg = cal_cfg.get("camera", {})

    pixel_path = project_root / pixel_cfg.get("persist_path", "data/calibration/pixel_calibration.yaml")
    camera_path = project_root / cam_cfg.get("persist_path", "data/calibration/camera_calibration.yaml")

    pixel = load_pixel_calibration(pixel_path)
    if pixel is None:
        known = float(pixel_cfg.get("known_distance_mm", 0.0) or 0.0)
        measured = float(pixel_cfg.get("measured_distance_px", 0.0) or 0.0)
        candidate = PixelCalibration(known, measured)
        pixel = candidate if candidate.is_valid else None

    camera = load_camera_calibration(camera_path)
    return CalibrationState(pixel=pixel, camera=camera, source=str(pixel_path))


def calibrate_camera_from_images(
    images: list[np.ndarray],
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
) -> CameraCalibration:
    """Run ``cv2.calibrateCamera`` on checkerboard frames."""
    if len(images) < 3:
        raise CalibrationError("Need at least 3 checkerboard images for camera calibration.")
    if board_cols < 2 or board_rows < 2:
        raise CalibrationError("Checkerboard inner-corner size is invalid.")
    if square_size_mm <= 0:
        raise CalibrationError("Checkerboard square size must be > 0 mm.")

    pattern_size = (int(board_cols), int(board_rows))
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None
    used = 0

    for image in images:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            continue
        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(objp.copy())
        image_points.append(refined)
        used += 1

    if used < 3 or image_size is None:
        raise CalibrationError(
            f"Checkerboard detected in only {used} image(s). Need at least 3 valid views."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    logger.info("Camera calibrated from %d views, RMS=%.4f px", used, rms)
    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_error=float(rms),
        checkerboard=pattern_size,
        square_size_mm=float(square_size_mm),
    )


def undistort_image(image: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Correct lens distortion using the saved camera model."""
    h, w = image.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera.camera_matrix,
        camera.dist_coeffs,
        (w, h),
        1,
        (w, h),
    )
    undistorted = cv2.undistort(image, camera.camera_matrix, camera.dist_coeffs, None, new_mtx)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y : y + rh, x : x + rw]
    return undistorted


def apply_camera_undistort(image: np.ndarray, state: CalibrationState, enabled: bool) -> np.ndarray:
    if not enabled or state.camera is None:
        return image
    try:
        return undistort_image(image, state.camera)
    except cv2.error as exc:
        logger.warning("Undistort failed, using original image: %s", exc)
        return image


def require_production_calibration(state: CalibrationState, config: dict[str, Any]) -> None:
    required = bool(config.get("calibration", {}).get("require_for_production", True))
    if required and not state.is_production_ready:
        raise CalibrationError(
            "Production measurement is blocked: calibration status is NOT CALIBRATED. "
            "Enter a known mm distance and the matching pixel length."
        )
