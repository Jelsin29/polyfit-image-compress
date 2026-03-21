"""Visualization utilities for compression comparison and analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from polyfit_compress.metrics import mse, psnr, ssim  # noqa: E402


def compare_images(
    original: NDArray[np.uint8],
    reconstructed: NDArray[np.uint8],
    title: str = "Compression Comparison",
    metrics_overlay: bool = True,
) -> Figure:
    """Create side-by-side comparison plot of original and reconstructed images.

    Parameters
    ----------
    original : NDArray[np.uint8]
        Original image array (2D grayscale or 3D RGB).
    reconstructed : NDArray[np.uint8]
        Reconstructed image array (must match original shape).
    title : str
        Plot super-title.
    metrics_overlay : bool
        If True, compute and display PSNR, SSIM, and MSE on the figure.

    Returns
    -------
    Figure
        The comparison figure. Caller decides whether to show or save.
    """
    is_grayscale = original.ndim == 2
    cmap = "gray" if is_grayscale else None

    fig, (ax_orig, ax_recon) = plt.subplots(1, 2, figsize=(12, 5))

    ax_orig.imshow(original, cmap=cmap)
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    ax_recon.imshow(reconstructed, cmap=cmap)
    ax_recon.set_title("Reconstructed")
    ax_recon.axis("off")

    fig.suptitle(title, fontsize=14)

    if metrics_overlay:
        psnr_val = psnr(original, reconstructed)
        ssim_val = ssim(original, reconstructed)
        mse_val = mse(original, reconstructed)

        psnr_text = f"PSNR: {psnr_val:.2f} dB" if psnr_val != float("inf") else "PSNR: inf dB"
        metrics_text = f"{psnr_text}  |  SSIM: {ssim_val:.4f}  |  MSE: {mse_val:.2f}"

        fig.text(
            0.5,
            0.02,
            metrics_text,
            ha="center",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig


def error_heatmap(
    original: NDArray[np.uint8],
    reconstructed: NDArray[np.uint8],
    title: str = "Error Heatmap",
) -> Figure:
    """Create heatmap showing pixel-wise error between original and reconstructed.

    For RGB images the absolute difference is averaged across the channel axis
    to produce a single-channel error map.

    Parameters
    ----------
    original : NDArray[np.uint8]
        Original image array (2D grayscale or 3D RGB).
    reconstructed : NDArray[np.uint8]
        Reconstructed image array (must match original shape).
    title : str
        Plot title.

    Returns
    -------
    Figure
        The error heatmap figure. Caller decides whether to show or save.
    """
    diff = np.abs(original.astype(np.float64) - reconstructed.astype(np.float64))

    # For multi-channel images, average across channels
    if diff.ndim == 3:
        diff = np.mean(diff, axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    im = ax.imshow(diff, cmap="inferno")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    fig.colorbar(im, ax=ax, label="Absolute Error")

    min_err = float(np.min(diff))
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))

    stats_text = f"Min: {min_err:.2f}  |  Max: {max_err:.2f}  |  Mean: {mean_err:.2f}"
    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig
