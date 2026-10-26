"""Inspection records → CSV / Excel."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .measurement import InspectionResult

logger = logging.getLogger(__name__)


def _pandas():
    """Import pandas lazily so measurement still works if Excel deps are blocked."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas/openpyxl could not be imported. Excel export is unavailable "
            f"in this environment ({exc}). CSV export still works."
        ) from exc
    return pd

COLUMNS = [
    "timestamp",
    "image",
    "calibration_factor_mm_per_px",
    "calibration_status",
    "L1_width_mm",
    "L1_height_mm",
    "L1_width_px",
    "L1_height_px",
    "L1_angle_deg",
    "L2_width_mm",
    "L2_height_mm",
    "L2_width_px",
    "L2_height_px",
    "L2_angle_deg",
    "distance_c2c_mm",
    "distance_edge_mm",
    "distance_c2c_px",
    "distance_edge_px",
    "overall_width_mm",
    "overall_height_mm",
    "result",
    "status",
    "message",
    "confidence",
    "pair_score",
    "candidate_count",
]


def result_to_record(result: InspectionResult, image_name: str = "") -> dict[str, Any]:
    factor: Optional[float] = None
    if result.calibration.is_production_ready:
        factor = float(result.calibration.mm_per_pixel)
    primary = result.leg1 or result.object
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": image_name or result.image_name,
        "calibration_factor_mm_per_px": factor,
        "calibration_status": result.calibration.status,
        "L1_width_mm": None if primary is None else primary.width_mm,
        "L1_height_mm": None if primary is None else primary.height_mm,
        "L1_width_px": None if primary is None else primary.width_px,
        "L1_height_px": None if primary is None else primary.height_px,
        "L1_angle_deg": None if primary is None else primary.angle_deg,
        "L2_width_mm": None if result.leg2 is None else result.leg2.width_mm,
        "L2_height_mm": None if result.leg2 is None else result.leg2.height_mm,
        "L2_width_px": None if result.leg2 is None else result.leg2.width_px,
        "L2_height_px": None if result.leg2 is None else result.leg2.height_px,
        "L2_angle_deg": None if result.leg2 is None else result.leg2.angle_deg,
        "distance_c2c_mm": result.center_distance_mm,
        "distance_edge_mm": result.edge_distance_mm,
        "distance_c2c_px": result.center_distance_px,
        "distance_edge_px": result.edge_distance_px,
        "overall_width_mm": result.overall_width_mm,
        "overall_height_mm": result.overall_height_mm,
        "result": result.overall_result,
        "status": result.status,
        "message": result.message,
        "confidence": result.confidence,
        "pair_score": result.pair_score,
        "candidate_count": result.candidate_count,
    }


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def save_csv(result: InspectionResult, path: str | Path, image_name: str = "") -> Path:
    """Append one inspection row using the standard library (no pandas)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = result_to_record(result, image_name)
    new_file = not target.is_file()
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({key: _stringify(record.get(key)) for key in COLUMNS})
    logger.info("Wrote inspection record to %s", target)
    return target


def save_excel(result: InspectionResult, path: str | Path, image_name: str = "") -> Path:
    pd = _pandas()
    target = Path(path)
    if target.suffix.lower() not in {".xlsx", ".xls"}:
        target = target.with_suffix(".xlsx")
    target.parent.mkdir(parents=True, exist_ok=True)
    record = result_to_record(result, image_name)
    frame = pd.DataFrame([record], columns=COLUMNS)
    if target.is_file():
        existing = pd.read_excel(target)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_excel(target, index=False)
    logger.info("Wrote inspection record to %s", target)
    return target


def save_both(
    result: InspectionResult,
    config: dict[str, Any],
    project_root: Path,
    image_name: str = "",
) -> tuple[Path, Optional[Path]]:
    reporting = config.get("reporting", {})
    csv_path = save_csv(
        result,
        project_root / reporting.get("csv_path", "data/output/inspections.csv"),
        image_name,
    )
    xls_target = project_root / reporting.get("excel_path", "data/output/inspections.xlsx")
    try:
        return csv_path, save_excel(result, xls_target, image_name)
    except Exception as exc:  # noqa: BLE001 — Excel deps may be blocked on this PC
        logger.warning("Excel export skipped: %s", exc)
        return csv_path, None
