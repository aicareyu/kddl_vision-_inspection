"""Configurable preprocessing pipeline.

Stages are applied only when enabled in ``config.yaml``. The function returns
every intermediate image so debug mode can display them.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _odd_kernel(size: int) -> int:
    size = int(max(1, size))
    return size if size % 2 == 1 else size + 1


def assess_image_quality(gray: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Simple focus / contrast checks. Not a substitute for optical validation."""
    qcfg = config.get("quality", {})
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    min_contrast = float(qcfg.get("min_contrast_std", 8.0))
    min_sharp = float(qcfg.get("min_sharpness", 15.0))
    poor = contrast < min_contrast or sharpness < min_sharp
    reasons: list[str] = []
    if contrast < min_contrast:
        reasons.append(f"low contrast (std={contrast:.1f})")
    if sharpness < min_sharp:
        reasons.append(f"low sharpness (var={sharpness:.1f})")
    if mean < 15:
        reasons.append("image is very dark")
        poor = True
    if mean > 245:
        reasons.append("image is saturated")
        poor = True
    return {
        "contrast_std": contrast,
        "sharpness": sharpness,
        "mean": mean,
        "poor": poor,
        "message": "; ".join(reasons) if reasons else "acceptable",
    }


def apply_blur(gray: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    if not cfg.get("enabled", False):
        return gray
    method = str(cfg.get("method", "gaussian")).lower()
    k = _odd_kernel(int(cfg.get("kernel_size", 3)))
    if method == "bilateral":
        return cv2.bilateralFilter(
            gray,
            int(cfg.get("bilateral_d", 7)),
            float(cfg.get("bilateral_sigma_color", 50)),
            float(cfg.get("bilateral_sigma_space", 50)),
        )
    sigma = float(cfg.get("sigma", 1.0))
    return cv2.GaussianBlur(gray, (k, k), sigma)


def apply_enhancement(gray: np.ndarray, pre_cfg: dict[str, Any]) -> np.ndarray:
    work = gray
    contrast_cfg = pre_cfg.get("contrast", {})
    if contrast_cfg.get("enabled", False):
        alpha = float(contrast_cfg.get("alpha", 1.2))
        beta = float(contrast_cfg.get("beta", 0))
        work = cv2.convertScaleAbs(work, alpha=alpha, beta=beta)

    clahe_cfg = pre_cfg.get("clahe", {})
    if clahe_cfg.get("enabled", False):
        clip = float(clahe_cfg.get("clip_limit", 2.0))
        tile = int(clahe_cfg.get("tile_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        work = clahe.apply(work)
    return work


def apply_threshold(gray: np.ndarray, seg_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(binary, edges)``. One of them is the primary mask depending on method."""
    method = str(seg_cfg.get("method", "adaptive")).lower()
    invert = bool(seg_cfg.get("invert_binary", False))
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    canny_cfg = seg_cfg.get("canny", {})
    edges = cv2.Canny(
        gray,
        int(canny_cfg.get("low", 50)),
        int(canny_cfg.get("high", 150)),
    )

    if method == "canny":
        binary = edges.copy()
        return binary, edges

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return binary, edges

    if method == "global":
        t = int(seg_cfg.get("global_threshold", 127))
        _, binary = cv2.threshold(gray, t, 255, thresh_type)
        return binary, edges

    # adaptive (default)
    adaptive_cfg = seg_cfg.get("adaptive", {})
    block = _odd_kernel(int(adaptive_cfg.get("block_size", 35)))
    if block < 3:
        block = 3
    c_val = float(adaptive_cfg.get("C", 7))
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if str(adaptive_cfg.get("method", "gaussian")).lower() == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    binary = cv2.adaptiveThreshold(gray, 255, adaptive_method, thresh_type, block, c_val)
    return binary, edges


def maybe_resize(image: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
    if not cfg.get("enabled", False):
        return image, 1.0
    max_dim = int(cfg.get("max_dimension", 2000))
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    logger.info("Resized image from %sx%s to %sx%s (scale=%.4f)", w, h, new_size[0], new_size[1], scale)
    return resized, scale


def preprocess(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Run the configured preprocessing chain.

    Returns a dict of named stage images plus quality metrics. The key
    ``binary`` is the image handed to segmentation.
    """
    pre_cfg = config.get("preprocessing", {})
    seg_cfg = config.get("segmentation", {})

    work, scale = maybe_resize(image, pre_cfg.get("resize", {}))
    stages: dict[str, Any] = {"original": image, "working": work, "resize_scale": scale}

    gray = to_gray(work)
    if pre_cfg.get("invert", False):
        gray = cv2.bitwise_not(gray)
    stages["grayscale"] = gray
    stages["quality"] = assess_image_quality(gray, config)

    blurred = apply_blur(gray, pre_cfg.get("blur", {}))
    stages["blurred"] = blurred

    enhanced = apply_enhancement(blurred, pre_cfg)
    stages["enhanced"] = enhanced

    binary, edges = apply_threshold(enhanced, seg_cfg)
    stages["threshold"] = binary
    stages["edges"] = edges
    stages["binary"] = binary
    return stages
