"""Quantum feature maps for image block encoding."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.quantum import require_quantum


def quantum_design_matrix(block_size: int, n_qubits: int = 4) -> NDArray[np.float64]:
    """Build design matrix using quantum feature map expectation values."""
    require_quantum()
    raise NotImplementedError
