"""Offline smoke test: demo image + inspection (no Streamlit, no pandas)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import PROJECT_ROOT, load_config
from src.calibration import CalibrationState, PixelCalibration
from src.image_acquisition import create_demo_watch_hand
from src.measurement import run_inspection


def main() -> None:
    image, truth = create_demo_watch_hand()
    out = PROJECT_ROOT / "data" / "input" / "demo_watch_hand.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(str(out))
    print(f"wrote {out} {image.shape}")

    config = load_config()
    config["measurement"]["mode"] = "two_legs"
    config["measurement"]["subpixel"]["enabled"] = False
    state = CalibrationState(pixel=PixelCalibration(1.0, 1.0 / truth["mm_per_pixel"]))
    result = run_inspection(image, config, state, image_name=out.name)
    print("status", result.status, result.overall_result)
    print("message", result.message)
    assert result.leg1 is not None and result.leg2 is not None
    print("L1 mm", result.leg1.width_mm, result.leg1.height_mm)
    print("L2 mm", result.leg2.width_mm, result.leg2.height_mm)
    print("c2c mm", result.center_distance_mm, "edge mm", result.edge_distance_mm)
    print("truth", truth)
    overlay = PROJECT_ROOT / "data" / "output" / "demo_overlay.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    if result.overlay is not None:
        cv2.imencode(".png", result.overlay)[1].tofile(str(overlay))
        print("overlay", overlay)


if __name__ == "__main__":
    main()
