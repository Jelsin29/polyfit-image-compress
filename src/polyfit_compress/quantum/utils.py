"""Quantum utility functions for state preparation and coefficient extraction."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def normalize_state(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a vector to unit length for quantum state preparation.

    Parameters
    ----------
    vector : NDArray[np.float64]
        Input vector.

    Returns
    -------
    NDArray[np.float64]
        Unit-normalized vector. Returns zeros if input is a zero vector.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-15:
        return np.zeros_like(vector)
    return vector / norm


def prepare_b_state(b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prepare the observation vector b for quantum encoding.

    Pads to next power of 2 and normalizes.

    Parameters
    ----------
    b : NDArray[np.float64]
        Observation vector (flattened pixel block).

    Returns
    -------
    NDArray[np.float64]
        Padded and normalized state vector.
    """
    size = len(b)
    next_pow2 = 1 << math.ceil(math.log2(max(size, 1)))
    padded = np.zeros(next_pow2, dtype=np.float64)
    padded[:size] = b
    return normalize_state(padded)


def extract_coefficients(
    statevector: NDArray[np.complex128],
    num_coefficients: int,
    norm: float,
) -> NDArray[np.float64]:
    """Extract real coefficients from a quantum statevector.

    Parameters
    ----------
    statevector : NDArray[np.complex128]
        Full statevector from the quantum circuit.
    num_coefficients : int
        Number of polynomial coefficients to extract.
    norm : float
        Original norm of b vector for denormalization.

    Returns
    -------
    NDArray[np.float64]
        Real-valued polynomial coefficients.

    Notes
    -----
    The returned coefficients are a **directional approximation** — they point
    in the same direction as the classical least-squares solution but their
    magnitude may differ due to the statevector normalization constraint.
    Downstream consumers should be aware that the scale is approximate.
    """
    raw = np.real(statevector[:num_coefficients])
    return raw * norm


def calculate_n_qubits(system_size: int) -> int:
    """Calculate number of qubits needed for a system of given size.

    Parameters
    ----------
    system_size : int
        Size of the linear system (number of equations).

    Returns
    -------
    int
        Number of qubits (ceil(log2(system_size))), minimum 1.
    """
    if system_size <= 1:
        return 1
    return math.ceil(math.log2(system_size))
