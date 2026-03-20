"""Quality metrics for image compression evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def psnr(original: NDArray, compressed: NDArray, data_range: int = 255) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    raise NotImplementedError


def mse(original: NDArray, compressed: NDArray) -> float:
    """Calculate Mean Squared Error."""
    raise NotImplementedError


def ssim(original: NDArray, compressed: NDArray) -> float:
    """Calculate Structural Similarity Index."""
    raise NotImplementedError


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Calculate compression ratio."""
    raise NotImplementedError
