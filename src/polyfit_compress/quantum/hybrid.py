"""Hybrid classical-quantum compression pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from skimage.util import view_as_blocks

from polyfit_compress.compressor import CompressionResult, LeastSquaresCompressor
from polyfit_compress.models import PolynomialModel
from polyfit_compress.quantum._guard import require_quantum
from polyfit_compress.quantum.vqls import VQLSConfig, VQLSSolver


class HybridCompressor:
    """Hybrid classical-quantum compressor.

    Wraps LeastSquaresCompressor and replaces the pseudoinverse solve step
    with a VQLS optimization loop. Falls back to classical solve when VQLS
    doesn't converge.

    Parameters
    ----------
    model : PolynomialModel
        Polynomial surface model (e.g. LinearModel, QuadraticModel).
    block_size : int
        Block size for tiling the image. Must be between 2 and 64.
    vqls_config : VQLSConfig | None
        VQLS solver configuration. Uses defaults if None.
    progress_callback : Callable[[int, int, bool], None] | None
        Optional callback invoked after each block is processed.
        Arguments are (block_index, total_blocks, converged).
    """

    def __init__(
        self,
        model: PolynomialModel,
        block_size: int = 4,
        vqls_config: VQLSConfig | None = None,
        progress_callback: Callable[[int, int, bool], None] | None = None,
    ) -> None:
        require_quantum()
        self._classical = LeastSquaresCompressor(model=model, block_size=block_size)
        self._solver = VQLSSolver(config=vqls_config)
        self._progress_callback = progress_callback

    @property
    def model(self) -> PolynomialModel:
        """The polynomial model used for fitting."""
        return self._classical.model

    @property
    def block_size(self) -> int:
        """Block size used for image tiling."""
        return self._classical.block_size

    def compress(self, image: NDArray) -> CompressionResult:
        """Compress image using VQLS for coefficient solving.

        Processes blocks sequentially (VQLS is per-block).
        Falls back to classical pseudoinverse on convergence failure.

        Parameters
        ----------
        image : NDArray
            Input image, grayscale (H, W) or RGB (H, W, 3).

        Returns
        -------
        CompressionResult
            Compression result with reconstructed image and metadata.
        """
        image = self._classical._validate_image(image)
        original_shape = image.shape

        if image.ndim == 3:
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

    def _compress_channel(
        self, channel: NDArray[np.float64]
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], tuple[int, int]]:
        """Compress a single grayscale channel using hybrid VQLS/classical solve.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float64], tuple[int, int]]
            Coefficients, reconstructed channel, and padded shape (height, width).
        """
        bs = self.block_size
        h, w = channel.shape

        # Pad to make dimensions divisible by block_size
        pad_h = (bs - h % bs) % bs
        pad_w = (bs - w % bs) % bs
        channel_padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
        new_h, new_w = channel_padded.shape

        # Split into blocks
        blocks = view_as_blocks(channel_padded, block_shape=(bs, bs))
        n_rows, n_cols = blocks.shape[:2]
        total_blocks = n_rows * n_cols

        design_matrix = self._classical.design_matrix
        pinv_matrix = self._classical.pinv_matrix

        # Process each block with VQLS, falling back to classical on failure
        num_coeffs = self.model.num_coefficients
        all_coeffs = np.zeros((num_coeffs, total_blocks), dtype=np.float64)

        block_idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                block_flat = blocks[r, c].flatten().astype(np.float64)

                # Try VQLS solve
                coeffs, converged = self._solver.solve(design_matrix, block_flat)

                if not converged:
                    # Fallback to classical pseudoinverse
                    coeffs = pinv_matrix @ block_flat

                all_coeffs[:, block_idx] = coeffs[:num_coeffs]

                if self._progress_callback is not None:
                    self._progress_callback(block_idx, total_blocks, converged)

                block_idx += 1

        # Reconstruct from coefficients
        reconstructed_flat = design_matrix @ all_coeffs
        reconstructed_blocks = reconstructed_flat.T.reshape(n_rows, n_cols, bs, bs)
        reconstructed_image = reconstructed_blocks.transpose(0, 2, 1, 3).reshape(new_h, new_w)

        # Crop to original size
        reconstructed_channel = reconstructed_image[:h, :w]

        return all_coeffs.astype(np.float32), reconstructed_channel, (new_h, new_w)
