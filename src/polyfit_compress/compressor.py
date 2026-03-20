"""Core image compression engine using least squares polynomial surface fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.models import PolynomialModel


@dataclass(frozen=True)
class CompressionResult:
    """Result of image compression."""

    reconstructed: NDArray[np.uint8]
    coefficients: NDArray[np.float32]
    compressed_size_bytes: int
    original_size_bytes: int
    block_size: int
    model_name: str

    @property
    def compression_ratio(self) -> float:
        raise NotImplementedError


class LeastSquaresCompressor:
    """Image compressor using polynomial surface fitting via least squares."""

    def __init__(self, model: PolynomialModel, block_size: int = 8) -> None:
        raise NotImplementedError

    def compress(self, image: NDArray) -> CompressionResult:
        """Compress an image. Accepts grayscale (H,W) or RGB (H,W,3)."""
        raise NotImplementedError

    def _compress_channel(self, channel: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
        """Compress a single grayscale channel."""
        raise NotImplementedError

    def _validate_image(self, image: NDArray) -> NDArray:
        """Validate and normalize input image."""
        raise NotImplementedError
