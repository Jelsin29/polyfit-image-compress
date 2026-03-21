"""Quantum computing module for polyfit-image-compress (optional)."""

from __future__ import annotations

try:
    import pennylane as qml  # noqa: F401

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


def require_quantum() -> None:
    """Raise ImportError if PennyLane is not installed."""
    if not QUANTUM_AVAILABLE:
        raise ImportError(
            "Quantum features require PennyLane. "
            "Install with: pip install polyfit-image-compress[quantum]"
        )


from polyfit_compress.quantum.utils import (  # noqa: E402
    calculate_n_qubits,
    extract_coefficients,
    normalize_state,
    prepare_b_state,
)
from polyfit_compress.quantum.vqls import VQLSConfig, VQLSSolver  # noqa: E402

__all__ = [
    "QUANTUM_AVAILABLE",
    "require_quantum",
    "calculate_n_qubits",
    "extract_coefficients",
    "normalize_state",
    "prepare_b_state",
    "VQLSConfig",
    "VQLSSolver",
]
