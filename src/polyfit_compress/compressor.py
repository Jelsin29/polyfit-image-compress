"""Core image compression engine using least squares polynomial surface fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from skimage.util import view_as_blocks

from polyfit_compress.exceptions import InvalidBlockSizeError, UnsupportedImageFormatError
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
        return self.original_size_bytes / self.compressed_size_bytes


class LeastSquaresCompressor:
    """Image compressor using polynomial surface fitting via least squares."""

    def __init__(self, model: PolynomialModel, block_size: int = 8) -> None:
        if not isinstance(block_size, int):
            raise InvalidBlockSizeError(
                f"block_size must be an integer, got {type(block_size).__name__}"
            )
        if block_size < 2 or block_size > 64:
            raise InvalidBlockSizeError(
                f"block_size must be between 2 and 64, got {block_size}"
            )

        self.model = model
        self.block_size = block_size
        self.design_matrix = model.build_design_matrix(block_size)
        self.pinv_matrix = linalg.pinv(self.design_matrix)

    def compress(self, image: NDArray) -> CompressionResult:
        """Compress an image. Accepts grayscale (H,W) or RGB (H,W,3)."""
        image = self._validate_image(image)

        if image.ndim == 3:
            # RGB: process each channel independently
            channel_coeffs = []
            channel_reconstructed = []
            for c in range(3):
                coeffs, reconstructed = self._compress_channel(image[:, :, c])
                channel_coeffs.append(coeffs)
                channel_reconstructed.append(reconstructed)
            all_coeffs = np.stack(channel_coeffs, axis=0)
            reconstructed_image = np.stack(channel_reconstructed, axis=-1)
            original_size_bytes = image.shape[0] * image.shape[1] * 3
        else:
            # Grayscale
            all_coeffs, reconstructed_image = self._compress_channel(image)
            original_size_bytes = image.shape[0] * image.shape[1]

        final_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        compressed_size_bytes = all_coeffs.size * 4  # float32 = 4 bytes

        return CompressionResult(
            reconstructed=final_image,
            coefficients=all_coeffs,
            compressed_size_bytes=compressed_size_bytes,
            original_size_bytes=original_size_bytes,
            block_size=self.block_size,
            model_name=self.model.name,
        )

    def _compress_channel(self, channel: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
        """Compress a single grayscale channel."""
        h, w = channel.shape

        # Pad to make dimensions divisible by block_size
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        channel_padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")

        new_h, new_w = channel_padded.shape

        # Split into blocks
        blocks = view_as_blocks(channel_padded, block_shape=(self.block_size, self.block_size))
        n_rows, n_cols = blocks.shape[:2]

        # Flatten blocks for vectorized fitting
        blocks_flat = blocks.reshape(-1, self.block_size * self.block_size).T

        # Compute coefficients: pinv(A) @ b
        coeffs = self.pinv_matrix @ blocks_flat

        # Reconstruct: A @ coeffs
        reconstructed_flat = self.design_matrix @ coeffs

        # Reshape back to image
        reconstructed_blocks = reconstructed_flat.T.reshape(
            n_rows, n_cols, self.block_size, self.block_size
        )
        reconstructed_image = reconstructed_blocks.transpose(0, 2, 1, 3).reshape(
            new_h, new_w
        )

        # Crop to original size
        reconstructed_channel = reconstructed_image[:h, :w]

        return coeffs.astype(np.float32), reconstructed_channel

    def _validate_image(self, image: NDArray) -> NDArray:
        """Validate and normalize input image."""
        if image.ndim not in (2, 3):
            raise UnsupportedImageFormatError(
                f"Image must be 2D (grayscale) or 3D (RGB), got {image.ndim}D"
            )
        if image.ndim == 3 and image.shape[2] != 3:
            raise UnsupportedImageFormatError(
                f"3D image must have 3 channels (RGB), got {image.shape[2]}"
            )
        if image.dtype != np.float64:
            image = image.astype(np.float64)
        return image
