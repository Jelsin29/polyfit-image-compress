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
    original_shape: tuple[int, ...]
    padded_shape: tuple[int, int]

    @property
    def compression_ratio(self) -> float:
        return self.original_size_bytes / self.compressed_size_bytes


class LeastSquaresCompressor:
    """Image compressor using polynomial surface fitting via least squares."""

    def __init__(self, model: PolynomialModel, block_size: int = 8) -> None:
        if not isinstance(block_size, (int, np.integer)):
            raise InvalidBlockSizeError(
                f"block_size must be an integer, got {type(block_size).__name__}"
            )
        if block_size < 2 or block_size > 64:
            raise InvalidBlockSizeError(
                f"block_size must be between 2 and 64 (upper bound prevents "
                f"numerical instability in pseudoinverse), got {block_size}"
            )

        self.model = model
        self.block_size = block_size
        self.design_matrix = model.build_design_matrix(block_size)
        self.pinv_matrix = linalg.pinv(self.design_matrix)

    def compress(self, image: NDArray) -> CompressionResult:
        """Compress an image. Accepts grayscale (H,W) or RGB (H,W,3)."""
        image = self._validate_image(image)
        original_shape = image.shape

        if image.ndim == 3:
            # RGB: process each channel independently
            channel_coeffs = []
            channel_reconstructed = []
            padded = (0, 0)
            for c in range(3):
                coeffs, reconstructed, padded = self._compress_channel(image[:, :, c])
                channel_coeffs.append(coeffs)
                channel_reconstructed.append(reconstructed)
            all_coeffs = np.stack(channel_coeffs, axis=0)
            reconstructed_image = np.stack(channel_reconstructed, axis=-1)
            original_size_bytes = image.shape[0] * image.shape[1] * 3
        else:
            # Grayscale
            all_coeffs, reconstructed_image, padded = self._compress_channel(image)
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
            original_shape=original_shape,
            padded_shape=padded,
        )

    def decompress(
        self,
        coefficients: NDArray,
        original_shape: tuple[int, ...],
        padded_shape: tuple[int, int],
    ) -> NDArray[np.uint8]:
        """Reconstruct an image from coefficients.

        Parameters
        ----------
        coefficients : NDArray
            Coefficient array from compression. Shape is either
            ``(num_coeffs, num_blocks)`` for grayscale or
            ``(channels, num_coeffs, num_blocks)`` for RGB.
        original_shape : tuple[int, ...]
            Original image shape before padding.
        padded_shape : tuple[int, int]
            Image shape after padding (height, width).

        Returns
        -------
        NDArray[np.uint8]
            Reconstructed image clipped to [0, 255].

        Raises
        ------
        ValueError
            If coefficient dimensions don't match expected block layout.
        """
        pad_h, pad_w = padded_shape
        n_block_rows = pad_h // self.block_size
        n_block_cols = pad_w // self.block_size
        expected_blocks = n_block_rows * n_block_cols
        num_coeffs = self.model.num_coefficients

        if coefficients.ndim == 3:
            # RGB: (channels, num_coeffs, num_blocks)
            channels = coefficients.shape[0]
            if coefficients.shape[1] != num_coeffs or coefficients.shape[2] != expected_blocks:
                raise ValueError(
                    f"Coefficient shape {coefficients.shape} does not match expected "
                    f"({channels}, {num_coeffs}, {expected_blocks})"
                )
            channel_images = []
            for c in range(channels):
                channel_images.append(
                    self._reconstruct_channel(
                        coefficients[c], n_block_rows, n_block_cols, pad_h, pad_w
                    )
                )
            reconstructed = np.stack(channel_images, axis=-1)
        elif coefficients.ndim == 2:
            # Grayscale: (num_coeffs, num_blocks)
            if coefficients.shape[0] != num_coeffs or coefficients.shape[1] != expected_blocks:
                raise ValueError(
                    f"Coefficient shape {coefficients.shape} does not match expected "
                    f"({num_coeffs}, {expected_blocks})"
                )
            reconstructed = self._reconstruct_channel(
                coefficients, n_block_rows, n_block_cols, pad_h, pad_w
            )
        else:
            raise ValueError(f"Coefficients must be 2D or 3D, got {coefficients.ndim}D")

        # Crop to original shape
        h, w = original_shape[0], original_shape[1]
        reconstructed = reconstructed[:h, :w]

        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def _reconstruct_channel(
        self,
        coeffs: NDArray,
        n_block_rows: int,
        n_block_cols: int,
        pad_h: int,
        pad_w: int,
    ) -> NDArray[np.float64]:
        """Reconstruct a single channel from coefficients.

        Parameters
        ----------
        coeffs : NDArray
            Coefficient matrix of shape ``(num_coeffs, num_blocks)``.
        n_block_rows : int
            Number of block rows in the padded image.
        n_block_cols : int
            Number of block columns in the padded image.
        pad_h : int
            Padded image height.
        pad_w : int
            Padded image width.

        Returns
        -------
        NDArray[np.float64]
            Reconstructed channel of shape ``(pad_h, pad_w)``.
        """
        reconstructed_flat = self.design_matrix @ coeffs
        reconstructed_blocks = reconstructed_flat.T.reshape(
            n_block_rows, n_block_cols, self.block_size, self.block_size
        )
        return reconstructed_blocks.transpose(0, 2, 1, 3).reshape(pad_h, pad_w)

    def _compress_channel(
        self, channel: NDArray[np.float64]
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], tuple[int, int]]:
        """Compress a single grayscale channel.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float64], tuple[int, int]]
            Coefficients, reconstructed channel, and padded shape (height, width).
        """
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
        reconstructed_image = reconstructed_blocks.transpose(0, 2, 1, 3).reshape(new_h, new_w)

        # Crop to original size
        reconstructed_channel = reconstructed_image[:h, :w]

        return coeffs.astype(np.float32), reconstructed_channel, (new_h, new_w)

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
