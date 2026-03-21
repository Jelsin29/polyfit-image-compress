"""Tests for the core compressor module."""

from __future__ import annotations

import numpy as np
import pytest

from polyfit_compress.compressor import CompressionResult, LeastSquaresCompressor
from polyfit_compress.exceptions import InvalidBlockSizeError, UnsupportedImageFormatError
from polyfit_compress.models import LinearModel, QuadraticModel


class TestLeastSquaresCompressor:
    def test_compress_roundtrip_grayscale(self, sample_grayscale) -> None:
        compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
        result = compressor.compress(sample_grayscale)
        assert isinstance(result, CompressionResult)
        assert result.reconstructed.shape == sample_grayscale.shape
        assert result.reconstructed.dtype == np.uint8

    def test_compress_roundtrip_rgb(self, sample_rgb) -> None:
        compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
        result = compressor.compress(sample_rgb)
        assert result.reconstructed.shape == sample_rgb.shape
        assert result.reconstructed.dtype == np.uint8

    def test_solid_image_perfect_reconstruction(self, solid_image) -> None:
        """A solid color image should be perfectly reconstructed by any polynomial."""
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=8)
        result = compressor.compress(solid_image)
        np.testing.assert_array_equal(result.reconstructed, solid_image)

    def test_invalid_block_size_zero(self) -> None:
        with pytest.raises(InvalidBlockSizeError):
            LeastSquaresCompressor(model=LinearModel(), block_size=0)

    def test_invalid_block_size_negative(self) -> None:
        with pytest.raises(InvalidBlockSizeError):
            LeastSquaresCompressor(model=LinearModel(), block_size=-1)

    def test_invalid_block_size_too_large(self) -> None:
        with pytest.raises(InvalidBlockSizeError):
            LeastSquaresCompressor(model=LinearModel(), block_size=128)

    def test_invalid_block_size_one(self) -> None:
        with pytest.raises(InvalidBlockSizeError):
            LeastSquaresCompressor(model=LinearModel(), block_size=1)

    def test_invalid_block_size_65(self) -> None:
        with pytest.raises(InvalidBlockSizeError):
            LeastSquaresCompressor(model=LinearModel(), block_size=65)

    def test_valid_block_size_boundary_min(self) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=2)
        assert compressor.block_size == 2

    def test_valid_block_size_boundary_max(self) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=64)
        assert compressor.block_size == 64

    def test_numpy_int_block_size(self) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=np.int64(8))
        assert compressor.block_size == 8

    def test_invalid_block_size_float(self) -> None:
        with pytest.raises(InvalidBlockSizeError):
            LeastSquaresCompressor(model=LinearModel(), block_size=8.5)

    def test_unsupported_image_1d(self) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=8)
        with pytest.raises(UnsupportedImageFormatError):
            compressor.compress(np.zeros(100))

    def test_unsupported_image_4channel(self) -> None:
        compressor = LeastSquaresCompressor(model=LinearModel(), block_size=8)
        with pytest.raises(UnsupportedImageFormatError):
            compressor.compress(np.zeros((64, 64, 4)))

    def test_non_square_image(self) -> None:
        compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
        img = np.random.default_rng(42).integers(0, 256, size=(50, 70), dtype=np.uint8)
        result = compressor.compress(img)
        assert result.reconstructed.shape == (50, 70)

    def test_compression_ratio_positive(self, sample_grayscale) -> None:
        compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
        result = compressor.compress(sample_grayscale)
        assert result.compression_ratio > 0

    def test_compression_ratio_equals_expected(self, sample_grayscale) -> None:
        compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
        result = compressor.compress(sample_grayscale)
        expected = result.original_size_bytes / result.compressed_size_bytes
        assert result.compression_ratio == pytest.approx(expected)

    def test_linear_compresses_more_than_quadratic(self, sample_grayscale) -> None:
        linear_result = LeastSquaresCompressor(model=LinearModel(), block_size=8).compress(sample_grayscale)
        quad_result = LeastSquaresCompressor(model=QuadraticModel(), block_size=8).compress(sample_grayscale)
        assert linear_result.compression_ratio > quad_result.compression_ratio

    def test_tiny_image_single_block(self, tiny_image) -> None:
        compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
        result = compressor.compress(tiny_image)
        assert result.reconstructed.shape == tiny_image.shape
