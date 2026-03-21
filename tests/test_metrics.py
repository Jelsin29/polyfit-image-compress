"""Tests for quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from polyfit_compress.metrics import psnr, mse, ssim, compression_ratio


class TestMSE:
    def test_identical_images(self) -> None:
        img = np.full((8, 8), 128, dtype=np.uint8)
        assert mse(img, img) == 0.0

    def test_known_value(self) -> None:
        a = np.array([0, 0, 0, 0], dtype=np.float64)
        b = np.array([1, 1, 1, 1], dtype=np.float64)
        assert mse(a, b) == 1.0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            mse(np.zeros((8, 8)), np.zeros((8, 9)))


class TestPSNR:
    def test_identical_images_infinite(self) -> None:
        img = np.full((8, 8), 128, dtype=np.uint8)
        assert psnr(img, img) == float("inf")

    def test_known_value(self) -> None:
        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.full((8, 8), 10, dtype=np.uint8)
        # MSE = 100, PSNR = 10 * log10(255^2 / 100) ~ 28.13
        result = psnr(a, b)
        assert 28.0 < result < 28.2

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            psnr(np.zeros((8, 8)), np.zeros((8, 9)))


class TestSSIM:
    def test_identical_images(self) -> None:
        img = np.random.default_rng(42).integers(0, 256, size=(32, 32), dtype=np.uint8)
        assert ssim(img, img) == pytest.approx(1.0)

    def test_different_images_less_than_one(self) -> None:
        a = np.zeros((32, 32), dtype=np.uint8)
        b = np.full((32, 32), 255, dtype=np.uint8)
        assert ssim(a, b) < 1.0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            ssim(np.zeros((32, 32)), np.zeros((32, 33)))


class TestCompressionRatio:
    def test_basic(self) -> None:
        assert compression_ratio(1000, 200) == 5.0

    def test_no_compression(self) -> None:
        assert compression_ratio(100, 100) == 1.0

    def test_invalid_zero_compressed(self) -> None:
        with pytest.raises(ValueError):
            compression_ratio(100, 0)

    def test_invalid_negative_compressed(self) -> None:
        with pytest.raises(ValueError):
            compression_ratio(100, -1)
