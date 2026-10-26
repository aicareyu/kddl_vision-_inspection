"""Watch-hand leg measurement engine.

UI code lives in ``app.py``. All computer-vision and metrology logic lives here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from .calibration import CalibrationState
from .measurement import InspectionResult, run_inspection

__all__ = [
    "InspectionError",
    "InspectionResult",
    "load_config",
    "run_inspection",
    "PROJECT_ROOT",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InspectionError(Exception):
    """Operator-facing, non-fatal inspection error."""


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load ``config/config.yaml`` (or an explicit path)."""
    cfg_path = Path(path) if path is not None else PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.is_file():
        raise InspectionError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise InspectionError("Configuration file is empty or invalid.")
    return data


def configure_logging(config: Optional[dict[str, Any]] = None) -> None:
    """Configure process-wide logging once."""
    level_name = "INFO"
    if config:
        level_name = str(config.get("logging", {}).get("level", "INFO"))
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def empty_result(message: str, status: str = "FAILED") -> InspectionResult:
    """Build a failed inspection result without raising."""
    return InspectionResult(
        status=status,
        message=message,
        overall_result="NG",
        calibration=CalibrationState(),
    )


def as_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure an image is 8-bit for OpenCV drawing / Streamlit display."""
    if image.dtype == np.uint8:
        return image
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)
