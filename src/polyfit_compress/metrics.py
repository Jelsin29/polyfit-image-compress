"""Quality metrics for image compression evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity


def psnr(original: NDArray, compressed: NDArray, data_range: int = 255) -> float:
    """Calculate Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range² / MSE)

    Args:
        original: Original image array.
        compressed: Compressed image array (must match original shape).
        data_range: Maximum possible pixel value (default 255).

    Returns:
        PSNR value in dB, or float('inf') if images are identical.

    Raises:
        ValueError: If arrays have different shapes.
    """
    if original.shape != compressed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs compressed {compressed.shape}"
        )

    error = mse(original, compressed)
    if error == 0:
        return float("inf")

    return float(10.0 * np.log10((data_range ** 2) / error))


def mse(original: NDArray, compressed: NDArray) -> float:
    """Calculate Mean Squared Error.

    MSE = mean((original - compressed)²)

    Args:
        original: Original image array.
        compressed: Compressed image array (must match original shape).

    Returns:
        Mean squared error as a float.

    Raises:
        ValueError: If arrays have different shapes.
    """
    if original.shape != compressed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs compressed {compressed.shape}"
        )

    orig = original.astype(np.float64)
    comp = compressed.astype(np.float64)
    return float(np.mean((orig - comp) ** 2))


def ssim(original: NDArray, compressed: NDArray) -> float:
    """Calculate Structural Similarity Index.

    Uses skimage.metrics.structural_similarity internally.

    Args:
        original: Original image array.
        compressed: Compressed image array (must match original shape).

    Returns:
        SSIM value between -1 and 1 (1 means identical).

    Raises:
        ValueError: If arrays have different shapes.
    """
    if original.shape != compressed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs compressed {compressed.shape}"
        )

    # Determine if the image is multichannel (e.g. RGB with shape H x W x C)
    channel_axis: int | None = None
    if original.ndim == 3:
        channel_axis = -1

    return float(
        structural_similarity(
            original,
            compressed,
            data_range=255,
            channel_axis=channel_axis,
        )
    )


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Calculate compression ratio.

    Args:
        original_bytes: Size of original data in bytes.
        compressed_bytes: Size of compressed data in bytes.

    Returns:
        Ratio of original to compressed size.

    Raises:
        ValueError: If compressed_bytes is zero or negative.
    """
    if compressed_bytes <= 0:
        raise ValueError(
            f"compressed_bytes must be positive, got {compressed_bytes}"
        )

    return float(original_bytes / compressed_bytes)
