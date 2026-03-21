"""Visualization utilities for compression comparison and analysis."""

from __future__ import annotations

from numpy.typing import NDArray


def compare_images(
    original: NDArray,
    reconstructed: NDArray,
    title: str = "Compression Comparison",
) -> object:
    """Create side-by-side comparison plot of original and reconstructed images.

    Parameters
    ----------
    original : NDArray
        Original image array.
    reconstructed : NDArray
        Reconstructed image array.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The comparison figure.
    """
    raise NotImplementedError("Stub — implement in phase2/visualization")


def error_heatmap(
    original: NDArray,
    reconstructed: NDArray,
    title: str = "Error Heatmap",
) -> object:
    """Create heatmap showing pixel-wise error between original and reconstructed.

    Parameters
    ----------
    original : NDArray
        Original image array.
    reconstructed : NDArray
        Reconstructed image array.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The error heatmap figure.
    """
    raise NotImplementedError("Stub — implement in phase2/visualization")
