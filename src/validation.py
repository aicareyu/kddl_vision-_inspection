"""Configurable tolerance evaluation (OK / NG)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LimitCheck:
    name: str
    value: Optional[float]
    minimum: float
    maximum: float
    status: str

    @property
    def ok(self) -> bool:
        return self.status == "OK"


@dataclass
class ToleranceResult:
    checks: list[LimitCheck] = field(default_factory=list)
    overall: str = "NG"
    leg1: str = "NG"
    leg2: str = "NG"
    distance: str = "NG"

    object: str = "NG"

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "object": self.object,
            "leg1": self.leg1,
            "leg2": self.leg2,
            "distance": self.distance,
            "checks": [
                {
                    "name": c.name,
                    "value": c.value,
                    "min": c.minimum,
                    "max": c.maximum,
                    "status": c.status,
                }
                for c in self.checks
            ],
        }


def check_limit(name: str, value: Optional[float], minimum: float, maximum: float) -> LimitCheck:
    """Evaluate a single numeric measurement against [min, max]."""
    if value is None or not _finite(value):
        return LimitCheck(name, value, minimum, maximum, "NG")
    status = "OK" if minimum <= float(value) <= maximum else "NG"
    return LimitCheck(name, float(value), float(minimum), float(maximum), status)


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _group_status(checks: list[LimitCheck]) -> str:
    if not checks:
        return "NG"
    return "OK" if all(c.ok for c in checks) else "NG"


def evaluate_limits(
    *,
    leg1_width_mm: Optional[float],
    leg1_height_mm: Optional[float],
    leg2_width_mm: Optional[float],
    leg2_height_mm: Optional[float],
    distance_mm: Optional[float],
    config: dict[str, Any],
    leg1_angle_deg: Optional[float] = None,
    leg2_angle_deg: Optional[float] = None,
) -> ToleranceResult:
    """Pure tolerance check used by the engine and unit tests."""
    tol = config.get("tolerance", {})
    l1 = tol.get("leg_1", {})
    l2 = tol.get("leg_2", {})
    dist = tol.get("distance", {})
    ang = tol.get("angle_deg", {})

    checks = [
        check_limit("leg_1.width", leg1_width_mm, float(l1.get("width", {}).get("min", 0)), float(l1.get("width", {}).get("max", 0))),
        check_limit("leg_1.height", leg1_height_mm, float(l1.get("height", {}).get("min", 0)), float(l1.get("height", {}).get("max", 0))),
        check_limit("leg_2.width", leg2_width_mm, float(l2.get("width", {}).get("min", 0)), float(l2.get("width", {}).get("max", 0))),
        check_limit("leg_2.height", leg2_height_mm, float(l2.get("height", {}).get("min", 0)), float(l2.get("height", {}).get("max", 0))),
        check_limit("distance", distance_mm, float(dist.get("min", 0)), float(dist.get("max", 0))),
    ]
    if ang.get("enabled", False):
        checks.append(
            check_limit(
                "leg_1.angle",
                leg1_angle_deg,
                float(ang.get("min", -180)),
                float(ang.get("max", 180)),
            )
        )
        checks.append(
            check_limit(
                "leg_2.angle",
                leg2_angle_deg,
                float(ang.get("min", -180)),
                float(ang.get("max", 180)),
            )
        )

    result = ToleranceResult(checks=checks)
    result.leg1 = _group_status([c for c in checks if c.name.startswith("leg_1.")])
    result.leg2 = _group_status([c for c in checks if c.name.startswith("leg_2.")])
    result.distance = _group_status([c for c in checks if c.name == "distance"])
    result.overall = _group_status(checks)
    return result


def evaluate_object_limits(
    *,
    width: Optional[float],
    height: Optional[float],
    config: dict[str, Any],
) -> ToleranceResult:
    """OK/NG for a single object. Disabled tolerances pass when a value exists."""
    block = config.get("tolerance", {}).get("object", {})
    if not block.get("enabled", False):
        present = width is not None and height is not None
        status = "OK" if present else "NG"
        return ToleranceResult(overall=status, object=status, leg1="OK", leg2="OK", distance="OK")
    checks = [
        check_limit("object.width", width, float(block.get("width", {}).get("min", 0)), float(block.get("width", {}).get("max", 0))),
        check_limit("object.height", height, float(block.get("height", {}).get("min", 0)), float(block.get("height", {}).get("max", 0))),
    ]
    result = ToleranceResult(checks=checks)
    result.object = _group_status(checks)
    result.overall = result.object
    result.leg1 = "OK"
    result.leg2 = "OK"
    result.distance = "OK"
    return result


def evaluate_object_tolerances(result: Any, config: dict[str, Any]) -> ToleranceResult:
    obj = getattr(result, "object", None)
    if obj is None:
        return ToleranceResult(overall="NG", object="NG")
    if not bool(config.get("tolerance", {}).get("object", {}).get("enabled", False)):
        return evaluate_object_limits(width=obj.width_px, height=obj.height_px, config=config)
    return evaluate_object_limits(width=obj.width_mm, height=obj.height_mm, config=config)


def evaluate_tolerances(result: Any, config: dict[str, Any]) -> ToleranceResult:
    """Adapter used by ``run_inspection`` (duck-typed InspectionResult)."""
    if getattr(result, "mode", "") == "object" or (getattr(result, "object", None) is not None and result.leg2 is None):
        return evaluate_object_tolerances(result, config)
    if result.leg1 is None or result.leg2 is None:
        return ToleranceResult(overall="NG", leg1="NG", leg2="NG", distance="NG")
    if result.leg1.width_mm is None or result.center_distance_mm is None:
        return ToleranceResult(overall="OK", leg1="OK", leg2="OK", distance="OK")
    return evaluate_limits(
        leg1_width_mm=result.leg1.width_mm,
        leg1_height_mm=result.leg1.height_mm,
        leg2_width_mm=result.leg2.width_mm,
        leg2_height_mm=result.leg2.height_mm,
        distance_mm=result.center_distance_mm,
        config=config,
        leg1_angle_deg=result.leg1.angle_deg,
        leg2_angle_deg=result.leg2.angle_deg,
    )
