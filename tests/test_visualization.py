"""Tests for visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from polyfit_compress.visualization import compare_images, error_heatmap  # noqa: E402


@pytest.fixture
def grayscale_pair() -> tuple[np.ndarray, np.ndarray]:
    """Create a small grayscale original/reconstructed pair."""
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    noise = rng.integers(-10, 11, size=(32, 32))
    reconstructed = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return original, reconstructed


@pytest.fixture
def rgb_pair() -> tuple[np.ndarray, np.ndarray]:
    """Create a small RGB original/reconstructed pair."""
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    reconstructed = np.clip(
        original.astype(np.int16) + rng.integers(-10, 11, size=(32, 32, 3)), 0, 255
    ).astype(np.uint8)
    return original, reconstructed


class TestCompareImages:
    """Tests for compare_images."""

    def test_compare_images_returns_figure(
        self, grayscale_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """compare_images should return a matplotlib Figure."""
        original, reconstructed = grayscale_pair
        fig = compare_images(original, reconstructed)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_compare_images_grayscale(self, grayscale_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """Grayscale images should produce a valid figure with 2 axes."""
        original, reconstructed = grayscale_pair
        fig = compare_images(original, reconstructed)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt_close(fig)

    def test_compare_images_rgb(self, rgb_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """RGB images should produce a valid figure with 2 axes."""
        original, reconstructed = rgb_pair
        fig = compare_images(original, reconstructed)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt_close(fig)

    def test_compare_images_with_metrics(
        self, grayscale_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """When metrics_overlay=True, figure text should contain PSNR and SSIM."""
        original, reconstructed = grayscale_pair
        fig = compare_images(original, reconstructed, metrics_overlay=True)
        texts = [t.get_text() for t in fig.texts]
        metrics_text = " ".join(texts)
        assert "PSNR" in metrics_text
        assert "SSIM" in metrics_text
        assert "MSE" in metrics_text
        plt_close(fig)

    def test_compare_images_without_metrics(
        self, grayscale_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """When metrics_overlay=False, no metrics text should be present."""
        original, reconstructed = grayscale_pair
        fig = compare_images(original, reconstructed, metrics_overlay=False)
        texts = [t.get_text() for t in fig.texts]
        metrics_text = " ".join(texts)
        assert "PSNR" not in metrics_text
        plt_close(fig)


class TestErrorHeatmap:
    """Tests for error_heatmap."""

    def test_error_heatmap_returns_figure(
        self, grayscale_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """error_heatmap should return a matplotlib Figure."""
        original, reconstructed = grayscale_pair
        fig = error_heatmap(original, reconstructed)
        assert isinstance(fig, Figure)
        plt_close(fig)

    def test_error_heatmap_grayscale(self, grayscale_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """Grayscale images should produce a valid heatmap figure."""
        original, reconstructed = grayscale_pair
        fig = error_heatmap(original, reconstructed)
        axes = fig.get_axes()
        # Main axis + colorbar axis
        assert len(axes) >= 1
        plt_close(fig)

    def test_error_heatmap_rgb(self, rgb_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """RGB images should produce a valid heatmap with channel-averaged error."""
        original, reconstructed = rgb_pair
        fig = error_heatmap(original, reconstructed)
        axes = fig.get_axes()
        assert len(axes) >= 1
        # Check stats text is present
        texts = [t.get_text() for t in fig.texts]
        stats_text = " ".join(texts)
        assert "Min" in stats_text
        assert "Max" in stats_text
        assert "Mean" in stats_text
        plt_close(fig)


def plt_close(fig: Figure) -> None:
    """Close a figure to avoid memory leaks in tests."""
    import matplotlib.pyplot as plt

    plt.close(fig)
