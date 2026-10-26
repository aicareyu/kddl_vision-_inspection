"""Geometry helpers used by measurement."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import (
    angular_difference_deg,
    clip_roi,
    euclidean_distance,
    min_area_rect_features,
    min_edge_distance,
    order_rect_dimensions,
)


def test_euclidean_distance() -> None:
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_angular_difference_wraps() -> None:
    assert angular_difference_deg(-5.0, 5.0) == pytest.approx(10.0)
    assert angular_difference_deg(-80.0, 80.0) == pytest.approx(20.0)
    assert angular_difference_deg(10.0, 190.0) == pytest.approx(0.0)


def test_order_rect_dimensions() -> None:
    width, height = order_rect_dimensions(40.0, 12.0)
    assert width == pytest.approx(12.0)
    assert height == pytest.approx(40.0)


def test_min_area_rect_axis_aligned() -> None:
    contour = np.array([[[10, 10]], [[30, 10]], [[30, 50]], [[10, 50]]], dtype=np.int32)
    feats = min_area_rect_features(contour)
    assert feats["width"] == pytest.approx(20.0, abs=0.5)
    assert feats["height"] == pytest.approx(40.0, abs=0.5)


def test_min_edge_distance_separated_squares() -> None:
    a = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    b = np.array([[[20, 0]], [[30, 0]], [[30, 10]], [[20, 10]]], dtype=np.int32)
    assert min_edge_distance(a, b) == pytest.approx(10.0, abs=0.1)


def test_clip_roi() -> None:
    x, y, w, h = clip_roi(-10, -10, 500, 500, (100, 80))
    assert (x, y) == (0, 0)
    assert w == 80
    assert h == 100
