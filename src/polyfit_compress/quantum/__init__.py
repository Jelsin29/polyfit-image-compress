"""Quantum computing module for polyfit-image-compress (optional)."""

from __future__ import annotations

try:
    import pennylane as qml
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
