"""PennyLane availability guard — imported by submodules to avoid circular imports."""

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
