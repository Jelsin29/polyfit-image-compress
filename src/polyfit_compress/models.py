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


class LinearModel:
    """Linear polynomial surface: z = c0 + c1*x + c2*y"""

    @property
    def num_coefficients(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def build_design_matrix(self, block_size: int) -> NDArray[np.float64]:
        raise NotImplementedError


class QuadraticModel:
    """Quadratic polynomial surface: z = c0 + c1*x + c2*y + c3*xy + c4*x² + c5*y²"""

    @property
    def num_coefficients(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def build_design_matrix(self, block_size: int) -> NDArray[np.float64]:
        raise NotImplementedError
