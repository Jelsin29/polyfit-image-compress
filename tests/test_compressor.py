"""Tests for the core compressor module."""

from __future__ import annotations


class TestLeastSquaresCompressor:
    """Test suite for LeastSquaresCompressor."""

    def test_compress_roundtrip_grayscale(self) -> None:
        raise NotImplementedError

    def test_compress_roundtrip_rgb(self) -> None:
        raise NotImplementedError

    def test_invalid_block_size_raises(self) -> None:
        raise NotImplementedError

    def test_non_square_image(self) -> None:
        raise NotImplementedError

    def test_compression_ratio_positive(self) -> None:
        raise NotImplementedError
