"""Binary cleanup and connected-component preparation."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def _kernel(size: int) -> np.ndarray:
    size = int(max(1, size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def apply_morphology(binary: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    if not cfg.get("enabled", False):
        return binary
    kernel = _kernel(int(cfg.get("kernel_size", 3)))
    work = binary.copy()
    operations = cfg.get("operations", ["close", "open"])
    op_map = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": None,
        "dilate": None,
        "erosion": None,
        "dilation": None,
    }
    for name in operations:
        key = str(name).lower()
        if key in {"erode", "erosion"}:
            work = cv2.erode(work, kernel, iterations=1)
        elif key in {"dilate", "dilation"}:
            work = cv2.dilate(work, kernel, iterations=1)
        elif key in op_map and op_map[key] is not None:
            work = cv2.morphologyEx(work, op_map[key], kernel)
        else:
            logger.warning("Unknown morphology operation: %s", name)
    return work


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a uint8 binary mask."""
    mask = binary > 0
    filled = ndimage.binary_fill_holes(mask)
    return (filled.astype(np.uint8)) * 255


def remove_small_components(binary: np.ndarray, min_area: float) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def segment(binary: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Morphological cleanup of the thresholded / edge image."""
    seg_cfg = config.get("segmentation", {})
    morph = apply_morphology(binary, seg_cfg.get("morphology", {}))
    if seg_cfg.get("fill_holes", True):
        morph = fill_holes(morph)
    min_area = float(seg_cfg.get("min_component_area", 20))
    cleaned = remove_small_components(morph, min_area)
    logger.debug("Segmentation kept components with area >= %.1f", min_area)
    return {
        "morphology": morph,
        "cleaned": cleaned,
        "binary": cleaned,
    }
