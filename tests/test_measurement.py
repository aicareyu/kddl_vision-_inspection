"""Pixel-to-mm conversion, box measurement, tolerances, and pipeline smoke test."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import CalibrationState, PixelCalibration
from src.geometry import min_area_rect_features
from src.image_acquisition import create_demo_object, create_demo_two_pins, create_demo_watch_hand
from src.measurement import pixels_to_mm, run_inspection
from src.report import save_csv
from src.validation import check_limit, evaluate_limits
from src import load_config


def test_pixels_to_mm() -> None:
    assert pixels_to_mm(50.0, 0.01) == pytest.approx(0.5)


def test_bounding_box_measurement() -> None:
    # 15 x 40 rectangle, rotated 0°.
    xs, ys = np.meshgrid(np.arange(10, 25), np.arange(20, 60))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2).astype(np.int32)
    feats = min_area_rect_features(pts)
    assert feats["width"] == pytest.approx(15.0, abs=1.0)
    assert feats["height"] == pytest.approx(40.0, abs=1.0)


def test_tolerance_ok_and_ng() -> None:
    config = {
        "tolerance": {
            "leg_1": {"width": {"min": 0.10, "max": 0.15}, "height": {"min": 0.20, "max": 0.30}},
            "leg_2": {"width": {"min": 0.10, "max": 0.15}, "height": {"min": 0.20, "max": 0.30}},
            "distance": {"min": 0.50, "max": 0.60},
        }
    }
    ok = evaluate_limits(
        leg1_width_mm=0.12,
        leg1_height_mm=0.25,
        leg2_width_mm=0.11,
        leg2_height_mm=0.24,
        distance_mm=0.55,
        config=config,
    )
    assert ok.overall == "OK"
    ng = evaluate_limits(
        leg1_width_mm=0.20,
        leg1_height_mm=0.25,
        leg2_width_mm=0.11,
        leg2_height_mm=0.24,
        distance_mm=0.55,
        config=config,
    )
    assert ng.overall == "NG"
    assert ng.leg1 == "NG"
    assert check_limit("x", None, 0.0, 1.0).status == "NG"


def test_save_csv_appends_record(tmp_path) -> None:
    image, truth = create_demo_watch_hand()
    config = load_config()
    config["measurement"]["mode"] = "two_legs"
    config["measurement"]["subpixel"]["enabled"] = False
    state = CalibrationState(pixel=PixelCalibration(1.0, 1.0 / truth["mm_per_pixel"]))
    result = run_inspection(image, config, state, image_name="csv_demo")
    path = tmp_path / "inspections.csv"
    save_csv(result, path, "csv_demo")
    save_csv(result, path, "csv_demo")
    text = path.read_text(encoding="utf-8")
    assert "L1_width_mm" in text
    assert text.count("csv_demo") == 2


def test_pipeline_blocks_without_calibration() -> None:
    image, _ = create_demo_watch_hand()
    config = load_config()
    config["measurement"]["mode"] = "two_legs"
    config["calibration"]["require_for_production"] = True
    result = run_inspection(image, config, CalibrationState(), image_name="nocal")
    assert result.status == "FAILED"
    assert "NOT CALIBRATED" in result.message


def test_demo_pipeline_detects_two_legs() -> None:
    image, truth = create_demo_watch_hand()
    config = load_config()
    config["debug"]["enabled"] = True
    config["measurement"]["mode"] = "two_legs"
    config["measurement"]["subpixel"]["enabled"] = False
    # Wide filters so the synthetic part is not rejected during unit tests.
    config["leg_detection"]["min_area"] = 20
    config["leg_detection"]["max_area"] = 20000
    config["tolerance"]["leg_1"]["width"] = {"min": 0.05, "max": 0.20}
    config["tolerance"]["leg_1"]["height"] = {"min": 0.10, "max": 0.40}
    config["tolerance"]["leg_2"]["width"] = {"min": 0.05, "max": 0.20}
    config["tolerance"]["leg_2"]["height"] = {"min": 0.10, "max": 0.40}
    config["tolerance"]["distance"] = {"min": 0.30, "max": 0.90}

    state = CalibrationState(
        pixel=PixelCalibration(1.0, 1.0 / truth["mm_per_pixel"]),
        source="test",
    )
    result = run_inspection(image, config, state, image_name="demo")
    assert result.status == "OK", result.message
    assert result.leg1 is not None and result.leg2 is not None
    assert result.center_distance_mm == pytest.approx(truth["distance_c2c_mm"], rel=0.25)
    assert result.leg1.width_mm == pytest.approx(truth["leg_width_mm"], rel=0.35)
    assert result.leg1.height_mm == pytest.approx(truth["leg_height_mm"], rel=0.35)
    assert result.overlay is not None


def test_two_pin_plate_reports_each_width() -> None:
    image, truth = create_demo_two_pins()
    config = load_config()
    config["debug"]["enabled"] = True
    config["measurement"]["mode"] = "two_legs"
    config["measurement"]["subpixel"]["enabled"] = False
    config["tolerance"]["leg_1"]["width"] = {"min": 0.05, "max": 0.30}
    config["tolerance"]["leg_1"]["height"] = {"min": 0.05, "max": 0.80}
    config["tolerance"]["leg_2"]["width"] = {"min": 0.05, "max": 0.30}
    config["tolerance"]["leg_2"]["height"] = {"min": 0.05, "max": 0.80}
    config["tolerance"]["distance"] = {"min": 1.0, "max": 8.0}
    state = CalibrationState(
        pixel=PixelCalibration(1.0, 1.0 / truth["mm_per_pixel"]),
        source="test",
    )
    result = run_inspection(image, config, state, image_name="two_pins")
    assert result.status == "OK", result.message
    assert result.leg1 is not None and result.leg2 is not None
    assert result.leg1.width_mm == pytest.approx(truth["leg_width_mm"], rel=1.6)
    assert result.leg2.width_mm == pytest.approx(truth["leg_width_mm"], rel=1.6)
    assert result.leg1.height_mm == pytest.approx(truth["leg_height_mm"], rel=0.55)
    assert result.leg2.height_mm == pytest.approx(truth["leg_height_mm"], rel=0.55)
    assert result.center_distance_mm == pytest.approx(truth["distance_c2c_mm"], rel=0.25)
    xs = sorted([result.leg1.centroid[0], result.leg2.centroid[0]])
    assert xs[1] - xs[0] > 200
    assert result.overlay is not None


def test_user_plate_gap_is_between_both_pins() -> None:
    from pathlib import Path

    from src.image_acquisition import load_image

    path = Path(__file__).resolve().parents[1] / "data" / "input" / "yellow_0.png"
    if not path.is_file():
        pytest.skip("user plate crop is not present")
    image = load_image(path)
    config = load_config()
    config["measurement"]["mode"] = "two_legs"
    config["measurement"]["subpixel"]["enabled"] = False
    result = run_inspection(image, config, CalibrationState(), image_name="user_plate")
    assert result.status == "OK", result.message
    assert result.leg1 is not None and result.leg2 is not None
    assert result.center_distance_px is not None
    assert result.center_distance_px > 60
    xs = sorted([result.leg1.centroid[0], result.leg2.centroid[0]])
    assert xs[1] - xs[0] > 40
    # Both real pins, not the near-right plate corner.
    width = float(image.shape[1])
    assert xs[0] < 0.45 * width
    assert xs[1] > 0.55 * width


def test_object_mode_measures_uploaded_shape() -> None:
    image, truth = create_demo_object()
    config = load_config()
    config["measurement"]["mode"] = "object"
    config["measurement"]["subpixel"]["enabled"] = False
    result = run_inspection(image, config, CalibrationState(), image_name="object")
    assert result.status == "OK", result.message
    assert result.object is not None
    assert result.object.width_px == pytest.approx(truth["width_px"], rel=0.15)
    assert result.object.height_px == pytest.approx(truth["height_px"], rel=0.15)
    assert result.object.width_mm is None
    assert result.overlay is not None
