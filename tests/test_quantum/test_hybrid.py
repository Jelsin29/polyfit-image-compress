"""Tests for hybrid classical-quantum pipeline."""

from __future__ import annotations

import numpy as np

from polyfit_compress.compressor import CompressionResult
from polyfit_compress.models import LinearModel
from polyfit_compress.quantum.vqls import VQLSConfig

from .conftest import quantum_required


@quantum_required
class TestHybridCompressor:
    """Tests for the HybridCompressor class."""

    def _make_compressor(
        self,
        block_size: int = 4,
        vqls_config: VQLSConfig | None = None,
        progress_callback=None,
    ):
        from polyfit_compress.quantum.hybrid import HybridCompressor

        return HybridCompressor(
            model=LinearModel(),
            block_size=block_size,
            vqls_config=vqls_config,
            progress_callback=progress_callback,
        )

    def test_hybrid_compressor_creates_result(self):
        """Compress a small image and verify CompressionResult is returned."""
        compressor = self._make_compressor(
            block_size=4,
            vqls_config=VQLSConfig(max_iterations=2, n_layers=1),
        )
        image = np.random.default_rng(0).integers(0, 256, (8, 8), dtype=np.uint8)
        result = compressor.compress(image)

        assert isinstance(result, CompressionResult)
        assert result.reconstructed.shape == (8, 8)
        assert result.reconstructed.dtype == np.uint8
        assert result.block_size == 4
        assert result.model_name == "linear"
        assert result.original_shape == (8, 8)
        assert result.compressed_size_bytes > 0
        assert result.original_size_bytes == 64

    def test_hybrid_compressor_grayscale(self):
        """Compress an 8x8 grayscale image with block_size=4."""
        compressor = self._make_compressor(
            block_size=4,
            vqls_config=VQLSConfig(max_iterations=2, n_layers=1),
        )
        image = np.random.default_rng(42).integers(0, 256, (8, 8), dtype=np.uint8)
        result = compressor.compress(image)

        assert result.reconstructed.shape == (8, 8)
        assert result.coefficients.ndim == 2
        # LinearModel has 3 coefficients, 4 blocks for 8x8 with block_size=4
        assert result.coefficients.shape == (3, 4)

    def test_hybrid_compressor_rgb(self):
        """Compress an 8x8 RGB image with block_size=4."""
        compressor = self._make_compressor(
            block_size=4,
            vqls_config=VQLSConfig(max_iterations=2, n_layers=1),
        )
        image = np.random.default_rng(7).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        result = compressor.compress(image)

        assert result.reconstructed.shape == (8, 8, 3)
        assert result.reconstructed.dtype == np.uint8
        # 3 channels, 3 coefficients, 4 blocks
        assert result.coefficients.shape == (3, 3, 4)
        assert result.original_size_bytes == 8 * 8 * 3

    def test_hybrid_compressor_fallback(self):
        """Verify fallback works when VQLS does not converge (max_iterations=1)."""
        callback_calls: list[tuple[int, int, bool]] = []

        def track_callback(idx: int, total: int, converged: bool) -> None:
            callback_calls.append((idx, total, converged))

        compressor = self._make_compressor(
            block_size=4,
            vqls_config=VQLSConfig(max_iterations=1, n_layers=1),
            progress_callback=track_callback,
        )
        image = np.random.default_rng(99).integers(0, 256, (8, 8), dtype=np.uint8)
        result = compressor.compress(image)

        # Should still produce a valid result via fallback
        assert isinstance(result, CompressionResult)
        assert result.reconstructed.shape == (8, 8)

        # With max_iterations=1, convergence is very unlikely
        # At least one block should have been processed
        assert len(callback_calls) == 4  # 8x8 / 4x4 = 4 blocks

    def test_hybrid_compressor_progress_callback(self):
        """Verify callback is called with correct arguments."""
        callback_calls: list[tuple[int, int, bool]] = []

        def track_callback(idx: int, total: int, converged: bool) -> None:
            callback_calls.append((idx, total, converged))

        compressor = self._make_compressor(
            block_size=4,
            vqls_config=VQLSConfig(max_iterations=2, n_layers=1),
            progress_callback=track_callback,
        )
        image = np.random.default_rng(1).integers(0, 256, (8, 8), dtype=np.uint8)
        compressor.compress(image)

        assert len(callback_calls) == 4
        for idx, (block_idx, total, converged) in enumerate(callback_calls):
            assert block_idx == idx
            assert total == 4
            assert isinstance(converged, bool)

    def test_hybrid_compressor_quality_parity(self):
        """PSNR should be within a reasonable range of the classical compressor."""
        from polyfit_compress.compressor import LeastSquaresCompressor

        model = LinearModel()
        image = np.random.default_rng(12).integers(50, 200, (8, 8), dtype=np.uint8)

        # Classical
        classical = LeastSquaresCompressor(model=model, block_size=4)
        classical_result = classical.compress(image)

        # Hybrid (with fallback likely due to low iterations)
        hybrid = self._make_compressor(
            block_size=4,
            vqls_config=VQLSConfig(max_iterations=2, n_layers=1),
        )
        hybrid_result = hybrid.compress(image)

        # Both should produce reconstructed images
        assert classical_result.reconstructed.shape == hybrid_result.reconstructed.shape

        # Compute PSNR for both
        def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
            mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
            if mse == 0:
                return float("inf")
            return float(10 * np.log10(255.0**2 / mse))

        classical_psnr = psnr(image, classical_result.reconstructed)
        hybrid_psnr = psnr(image, hybrid_result.reconstructed)

        # Hybrid should produce reasonable quality (at least 10 dB PSNR)
        assert hybrid_psnr > 10.0, f"Hybrid PSNR too low: {hybrid_psnr:.2f} dB"
        # And be in a reasonable range of classical (within 20 dB)
        assert abs(classical_psnr - hybrid_psnr) < 20.0, (
            f"Quality gap too large: classical={classical_psnr:.2f} dB, hybrid={hybrid_psnr:.2f} dB"
        )
