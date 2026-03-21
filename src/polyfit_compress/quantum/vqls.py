"""Variational Quantum Linear Solver for least squares coefficient computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.quantum import require_quantum


def solve_least_squares_quantum(
    design_matrix: NDArray[np.float64],
    block_pixels: NDArray[np.float64],
    n_layers: int = 4,
    max_iterations: int = 200,
) -> NDArray[np.float64]:
    """Solve least squares using Variational Quantum Linear Solver.

    Not yet implemented — coming in Phase 3.
    """
    require_quantum()
    raise NotImplementedError("VQLS solver is not yet implemented — coming in Phase 3")
