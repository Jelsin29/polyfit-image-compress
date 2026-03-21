"""Polynomial model definitions using Strategy pattern."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PolynomialModel(Protocol):
    """Protocol for polynomial surface models."""

    @property
    def num_coefficients(self) -> int: ...

    @property
    def name(self) -> str: ...

    def build_design_matrix(self, block_size: int) -> NDArray[np.float64]: ...


def _make_coords(block_size: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build flattened coordinate vectors for a block.

    Returns:
        Tuple of (ones, x_flat, y_flat) arrays, each of length block_size².
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    coords = np.linspace(0, block_size - 1, block_size)
    X, Y = np.meshgrid(coords, coords)
    x_flat: NDArray[np.float64] = X.flatten()
    y_flat: NDArray[np.float64] = Y.flatten()
    ones: NDArray[np.float64] = np.ones_like(x_flat)
    return ones, x_flat, y_flat


class LinearModel:
    """Linear polynomial surface: z = c0 + c1*x + c2*y"""

    @property
    def num_coefficients(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "linear"

    def build_design_matrix(self, block_size: int) -> NDArray[np.float64]:
        ones, x_flat, y_flat = _make_coords(block_size)
        return np.column_stack((ones, x_flat, y_flat))


class QuadraticModel:
    """Quadratic polynomial surface: z = c0 + c1*x + c2*y + c3*xy + c4*x² + c5*y²"""

    @property
    def num_coefficients(self) -> int:
        return 6

    @property
    def name(self) -> str:
        return "quadratic"

    def build_design_matrix(self, block_size: int) -> NDArray[np.float64]:
        ones, x_flat, y_flat = _make_coords(block_size)
        return np.column_stack((
            ones, x_flat, y_flat,
            x_flat * y_flat, x_flat**2, y_flat**2,
        ))
