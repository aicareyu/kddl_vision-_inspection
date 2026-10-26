"""Image acquisition abstraction.

Development uses stored files. Industrial cameras (Basler, Hikvision, IDS, FLIR)
plug in later by implementing ``CameraInterface`` — the measurement engine does
not change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class AcquisitionError(Exception):
    """Raised when an image cannot be acquired or decoded."""


class CameraInterface(ABC):
    """Vendor-agnostic camera / source interface."""

    @abstractmethod
    def connect(self) -> bool:
        """Open the device or file source. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the device or file source."""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """Return one BGR uint8 frame."""

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return a serialisable snapshot of source parameters."""

    def is_connected(self) -> bool:
        return False


class ImageFileCamera(CameraInterface):
    """Development camera that loads a single stored image."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._image: Optional[np.ndarray] = None
        self._connected = False

    def connect(self) -> bool:
        image = load_image(self.path)
        self._image = image
        self._connected = True
        logger.info("ImageFileCamera connected: %s shape=%s", self.path, image.shape)
        return True

    def disconnect(self) -> None:
        self._image = None
        self._connected = False

    def capture(self) -> np.ndarray:
        if not self._connected or self._image is None:
            raise AcquisitionError("ImageFileCamera is not connected.")
        return self._image.copy()

    def get_parameters(self) -> dict[str, Any]:
        h = w = c = 0
        if self._image is not None:
            h, w = self._image.shape[:2]
            c = 1 if self._image.ndim == 2 else self._image.shape[2]
        return {
            "backend": "file",
            "path": str(self.path),
            "width": w,
            "height": h,
            "channels": c,
            "connected": self._connected,
        }

    def is_connected(self) -> bool:
        return self._connected


class FolderCamera(CameraInterface):
    """Iterate images from a folder. ``capture()`` returns the current file."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self._files: list[Path] = []
        self._index = 0
        self._connected = False
        self._current: Optional[np.ndarray] = None

    def connect(self) -> bool:
        self._files = list_images(self.folder)
        if not self._files:
            raise AcquisitionError(f"No supported images in folder: {self.folder}")
        self._index = 0
        self._connected = True
        self._current = load_image(self._files[self._index])
        return True

    def disconnect(self) -> None:
        self._files = []
        self._current = None
        self._connected = False

    def capture(self) -> np.ndarray:
        if not self._connected or self._current is None:
            raise AcquisitionError("FolderCamera is not connected.")
        return self._current.copy()

    def select(self, index: int) -> np.ndarray:
        if not self._files:
            raise AcquisitionError("FolderCamera has no files loaded.")
        self._index = int(np.clip(index, 0, len(self._files) - 1))
        self._current = load_image(self._files[self._index])
        return self._current.copy()

    def get_parameters(self) -> dict[str, Any]:
        current = self._files[self._index] if self._files else None
        return {
            "backend": "folder",
            "folder": str(self.folder),
            "count": len(self._files),
            "index": self._index,
            "current": str(current) if current else "",
            "connected": self._connected,
        }

    def is_connected(self) -> bool:
        return self._connected

    @property
    def files(self) -> list[Path]:
        return list(self._files)


class _UnimplementedIndustrialCamera(CameraInterface):
    """Placeholder so future SDKs do not change the measurement pipeline."""

    vendor: str = "unknown"

    def connect(self) -> bool:
        raise NotImplementedError(
            f"{self.vendor} SDK integration is not implemented. "
            "Use ImageFileCamera during development."
        )

    def disconnect(self) -> None:
        return None

    def capture(self) -> np.ndarray:
        raise NotImplementedError(f"{self.vendor} capture is not implemented.")

    def get_parameters(self) -> dict[str, Any]:
        return {"backend": self.vendor.lower(), "implemented": False}


class BaslerCamera(_UnimplementedIndustrialCamera):
    vendor = "Basler"


class HikvisionCamera(_UnimplementedIndustrialCamera):
    vendor = "Hikvision"


class IDSCamera(_UnimplementedIndustrialCamera):
    vendor = "IDS"


class FLIRCamera(_UnimplementedIndustrialCamera):
    vendor = "FLIR"


def create_camera(backend: str, **kwargs: Any) -> CameraInterface:
    """Factory for camera backends listed in ``config.yaml``."""
    key = backend.strip().lower()
    mapping: dict[str, type[CameraInterface]] = {
        "file": ImageFileCamera,
        "folder": FolderCamera,
        "basler": BaslerCamera,
        "hikvision": HikvisionCamera,
        "ids": IDSCamera,
        "flir": FLIRCamera,
    }
    if key not in mapping:
        raise AcquisitionError(f"Unknown camera backend: {backend}")
    cls = mapping[key]
    if key == "file":
        return ImageFileCamera(kwargs.get("path", ""))
    if key == "folder":
        return FolderCamera(kwargs.get("folder", kwargs.get("path", "")))
    return cls()


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR uint8. Supports JPG, JPEG, PNG, BMP, TIFF."""
    file_path = Path(path)
    if not file_path.is_file():
        raise AcquisitionError(f"Image not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AcquisitionError(
            f"Unsupported image type '{file_path.suffix}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    data = np.fromfile(str(file_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise AcquisitionError(f"Invalid or unreadable image: {file_path}")
    if image.size == 0:
        raise AcquisitionError(f"Image is empty: {file_path}")
    logger.debug("Loaded %s shape=%s", file_path, image.shape)
    return image


def load_image_from_bytes(data: bytes, name: str = "upload") -> np.ndarray:
    """Decode an uploaded file from memory."""
    if not data:
        raise AcquisitionError("Uploaded file is empty.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise AcquisitionError(f"Invalid image upload: {name}")
    return image


def list_images(folder: str | Path) -> list[Path]:
    """Return supported image paths in ``folder`` (non-recursive)."""
    directory = Path(folder)
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def create_demo_watch_hand(
    width: int = 1200,
    height: int = 800,
    *,
    mm_per_pixel: float = 0.005,
) -> tuple[np.ndarray, dict[str, float]]:
    """Synthesize a backside-style watch-hand image with two tiny legs.

    Returns ``(bgr_image, ground_truth_mm)`` so the pipeline can be exercised
    without a production part image. Ground truth is approximate (anti-aliased
    rendering + morphology will shift edges by a fraction of a pixel).
    """
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 3.0, (height, width, 3))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    body_w, body_h = 520, 70
    cx, cy = width // 2, height // 2
    x0, y0 = cx - body_w // 2, cy - body_h // 2
    cv2.rectangle(image, (x0, y0), (x0 + body_w, y0 + body_h), (40, 40, 40), thickness=-1)
    cv2.ellipse(image, (x0 + 20, cy), (28, body_h // 2), 0, 90, 270, (40, 40, 40), -1)
    cv2.ellipse(image, (x0 + body_w - 8, cy), (90, 18), 0, -90, 90, (40, 40, 40), -1)

    # Two rectangular legs on the "back" (below the body), slightly rotated.
    leg_w_px, leg_h_px = 18, 46
    gap_px = 110
    leg_centers = (
        (cx - gap_px // 2, y0 + body_h + 40),
        (cx + gap_px // 2, y0 + body_h + 40),
    )
    angle = -8.0
    for (lx, ly) in leg_centers:
        rect = ((float(lx), float(ly)), (float(leg_w_px), float(leg_h_px)), angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(image, box, (35, 35, 35))

    truth = {
        "leg_width_mm": leg_w_px * mm_per_pixel,
        "leg_height_mm": leg_h_px * mm_per_pixel,
        "distance_c2c_mm": gap_px * mm_per_pixel,
        "mm_per_pixel": mm_per_pixel,
    }
    logger.info("Created demo watch-hand image %sx%s", width, height)
    return image, truth


def _fill_rounded_rect(image: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    radius = int(min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)


def create_demo_two_pins(
    width: int = 1280,
    height: int = 720,
    *,
    mm_per_pixel: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    """Synthesize a studio-style brass plate with two clear pins."""
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    rng = np.random.default_rng(5)
    image = np.clip(image.astype(np.float32) + rng.normal(0, 1.4, image.shape), 0, 255).astype(np.uint8)

    plate_w, plate_h = 620, 210
    cx, cy = width // 2, height // 2 + 20
    x0, y0 = cx - plate_w // 2, cy - plate_h // 2
    x1, y1 = x0 + plate_w, y0 + plate_h
    brass = (42, 158, 212)
    shadow = (208, 208, 208)

    plate = np.zeros_like(image)
    _fill_rounded_rect(plate, (x0 + 12, y0 + 18), (x1 + 12, y1 + 18), shadow, 28)
    _fill_rounded_rect(plate, (x0, y0), (x1, y1), brass, 28)
    grain = rng.normal(0, 4.0, image.shape)
    plate_mask = np.any(plate > 0, axis=2)
    textured = np.clip(plate.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    plate = np.where(plate_mask[..., None], textured, plate)

    pin_radius = 8
    stem = 30
    pin1 = (x0 + 130, cy + 8)
    pin2 = (x1 - 130, cy - 8)
    for px, py in (pin1, pin2):
        cv2.circle(plate, (px + 3, py + 4), pin_radius + 1, (28, 90, 120), -1)
        cv2.line(plate, (px, py), (px, py - stem), (32, 48, 58), 5, cv2.LINE_AA)
        cv2.circle(plate, (px, py), pin_radius, (28, 40, 48), -1)
        cv2.circle(plate, (px, py - stem), 3, (230, 236, 240), -1)

    angle = -12.0
    matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
    rotated = cv2.warpAffine(plate, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rot_mask = np.any(rotated > 8, axis=2)
    image[rot_mask] = rotated[rot_mask]

    def _rotate_point(point: tuple[int, int]) -> tuple[float, float]:
        vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        out = matrix @ vec
        return float(out[0]), float(out[1])

    r1 = _rotate_point(pin1)
    r2 = _rotate_point(pin2)
    dx, dy = r2[0] - r1[0], r2[1] - r1[1]
    truth = {
        "leg_width_mm": (2.0 * pin_radius) * mm_per_pixel,
        "leg_height_mm": (stem + pin_radius) * mm_per_pixel,
        "distance_c2c_mm": float(np.hypot(dx, dy) * mm_per_pixel),
        "mm_per_pixel": mm_per_pixel,
        "leg_width_px": float(2 * pin_radius),
        "leg_height_px": float(stem + pin_radius),
        "distance_c2c_px": float(np.hypot(dx, dy)),
    }
    logger.info("Created demo two-pin plate %sx%s", width, height)
    return image, truth


def create_demo_object(
    width: int = 1000,
    height: int = 700,
    *,
    obj_w: int = 260,
    obj_h: int = 140,
) -> tuple[np.ndarray, dict[str, float]]:
    """Synthesize a single dark object on a light field for object-mode tests."""
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    x0, y0 = cx - obj_w // 2, cy - obj_h // 2
    cv2.rectangle(image, (x0, y0), (x0 + obj_w, y0 + obj_h), (30, 30, 30), thickness=-1)
    return image, {"width_px": float(obj_w), "height_px": float(obj_h)}


def iter_supported_names(paths: Iterable[Path]) -> list[str]:
    return [p.name for p in paths]
