"""Shared test fixtures for polyfit-image-compress."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def sample_grayscale() -> NDArray[np.uint8]:
    """8-bit grayscale test image (64x64 gradient)."""
    return np.tile(np.arange(64, dtype=np.uint8), (64, 1))


@pytest.fixture
def sample_rgb() -> NDArray[np.uint8]:
    """8-bit RGB test image (64x64)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    img[:, :, 1] = np.tile(np.arange(64, dtype=np.uint8).reshape(64, 1), (1, 64))
    img[:, :, 2] = 128
    return img


@pytest.fixture
def solid_image() -> NDArray[np.uint8]:
    """Solid color image (should compress perfectly)."""
    return np.full((32, 32), 128, dtype=np.uint8)


@pytest.fixture
def tiny_image() -> NDArray[np.uint8]:
    """Minimal 8x8 image (single block)."""
    return np.random.default_rng(42).integers(0, 256, size=(8, 8), dtype=np.uint8)
