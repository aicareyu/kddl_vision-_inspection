"""Calibration: pixel scale, persistence, and camera-model I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.calibration import (
    CalibrationError,
    CalibrationState,
    CameraCalibration,
    PixelCalibration,
    compute_mm_per_pixel,
    load_camera_calibration,
    load_pixel_calibration,
    require_production_calibration,
    save_camera_calibration,
    save_pixel_calibration,
    undistort_image,
)


def test_compute_mm_per_pixel() -> None:
    assert compute_mm_per_pixel(2.0, 400.0) == pytest.approx(0.005)


def test_invalid_pixel_calibration() -> None:
    with pytest.raises(CalibrationError):
        compute_mm_per_pixel(1.0, 0.0)
    with pytest.raises(CalibrationError):
        compute_mm_per_pixel(0.0, 100.0)
    assert not PixelCalibration(1.0, 0.0).is_valid


def test_pixels_to_mm_roundtrip() -> None:
    state = CalibrationState(pixel=PixelCalibration(1.0, 200.0))
    assert state.status == "CALIBRATED"
    assert state.pixels_to_mm(200.0) == pytest.approx(1.0)
    assert state.mm_to_pixels(0.5) == pytest.approx(100.0)


def test_not_calibrated_blocks_production() -> None:
    state = CalibrationState()
    assert state.status == "NOT CALIBRATED"
    with pytest.raises(CalibrationError):
        require_production_calibration(state, {"calibration": {"require_for_production": True}})
    require_production_calibration(state, {"calibration": {"require_for_production": False}})


def test_pixel_calibration_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "pixel_calibration.yaml"
    save_pixel_calibration(PixelCalibration(1.25, 250.0), path)
    loaded = load_pixel_calibration(path)
    assert loaded is not None
    assert loaded.mm_per_pixel == pytest.approx(0.005)


def test_camera_calibration_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "camera_calibration.yaml"
    cam = CameraCalibration(
        camera_matrix=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros(5),
        image_size=(640, 480),
        rms_error=0.12,
    )
    save_camera_calibration(cam, path)
    loaded = load_camera_calibration(path)
    assert loaded is not None
    assert loaded.rms_error == pytest.approx(0.12)
    assert loaded.image_size == (640, 480)
    np.testing.assert_allclose(loaded.camera_matrix, cam.camera_matrix)

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    undistorted = undistort_image(frame, loaded)
    assert undistorted.size > 0
