"""Quantum test configuration."""

import pytest

try:
    import pennylane  # noqa: F401

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

quantum_required = pytest.mark.skipif(
    not HAS_PENNYLANE,
    reason="PennyLane not installed",
)
