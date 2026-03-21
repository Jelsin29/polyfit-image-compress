"""Quantum computing module for polyfit-image-compress (optional)."""

from __future__ import annotations

from polyfit_compress.quantum._guard import QUANTUM_AVAILABLE, require_quantum
from polyfit_compress.quantum.feature_maps import QuantumFeatureMap
from polyfit_compress.quantum.utils import (
    calculate_n_qubits,
    extract_coefficients,
    normalize_state,
    prepare_b_state,
)
from polyfit_compress.quantum.vqls import VQLSConfig, VQLSSolver

__all__ = [
    "QUANTUM_AVAILABLE",
    "QuantumFeatureMap",
    "require_quantum",
    "calculate_n_qubits",
    "extract_coefficients",
    "normalize_state",
    "prepare_b_state",
    "VQLSConfig",
    "VQLSSolver",
]
