"""Quantum feature maps for image block encoding."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.quantum._guard import require_quantum


class QuantumFeatureMap:
    """Quantum feature map that creates design matrix columns from quantum circuits.

    Instead of polynomial basis functions (1, x, y, xy, x², y²), uses
    expectation values of Pauli operators after encoding pixel coordinates
    as rotation angles.

    Parameters
    ----------
    n_features : int
        Number of features (qubits) for the feature map. Default is 6.
    device_name : str
        PennyLane device backend name. Default is ``"default.qubit"``.
    """

    def __init__(self, n_features: int = 6, device_name: str = "default.qubit") -> None:
        require_quantum()
        if n_features < 2:
            raise ValueError(f"n_features must be >= 2, got {n_features}")
        self._n_features = n_features
        self._device_name = device_name

    @property
    def num_coefficients(self) -> int:
        """Number of coefficients produced by this feature map."""
        return self._n_features

    @property
    def name(self) -> str:
        """Human-readable name for this model."""
        return f"quantum-feature-map-{self._n_features}"

    def build_design_matrix(self, block_size: int) -> NDArray[np.float64]:
        """Build design matrix using quantum feature map.

        For each pixel coordinate (x, y) in the block, encodes the position
        as RY rotations on each qubit, applies CNOT entanglement, and measures
        PauliZ expectation values to form feature columns.

        Parameters
        ----------
        block_size : int
            Side length of the image block.

        Returns
        -------
        NDArray[np.float64]
            Design matrix of shape (block_size², n_features).
        """
        import pennylane as qml

        n_qubits = self._n_features
        dev = qml.device(self._device_name, wires=n_qubits)

        @qml.qnode(dev)
        def feature_circuit(x: float, y: float) -> list[float]:
            """Encode (x, y) independently on alternating qubits, entangle, measure PauliZ."""
            for i in range(n_qubits):
                # Even qubits encode x, odd qubits encode y
                angle = x if i % 2 == 0 else y
                qml.RY(angle * np.pi / block_size, wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        coords = np.arange(block_size)
        xx, yy = np.meshgrid(coords, coords)
        x_flat = xx.flatten().astype(np.float64)
        y_flat = yy.flatten().astype(np.float64)

        n_pixels = block_size * block_size
        design_matrix = np.empty((n_pixels, n_qubits), dtype=np.float64)

        for idx in range(n_pixels):
            result = feature_circuit(x_flat[idx], y_flat[idx])
            design_matrix[idx, :] = np.array(result, dtype=np.float64)

        return design_matrix
