"""Variational Quantum Linear Solver for least squares coefficient computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from polyfit_compress.quantum._guard import require_quantum
from polyfit_compress.quantum.utils import (
    calculate_n_qubits,
    extract_coefficients,
    prepare_b_state,
)


@dataclass
class VQLSConfig:
    """Configuration for the VQLS solver.

    Attributes
    ----------
    n_layers : int
        Number of StronglyEntanglingLayers in the ansatz.
    max_iterations : int
        Maximum optimization iterations.
    stepsize : float
        Adam optimizer learning rate.
    convergence_threshold : float
        Cost value below which the solver considers the system converged.
    device_name : str
        PennyLane device name.
    """

    n_layers: int = 4
    max_iterations: int = 200
    stepsize: float = 0.1
    convergence_threshold: float = 1e-4
    device_name: str = "default.qubit"


class VQLSSolver:
    """Variational Quantum Linear Solver.

    Solves the linear system Ax = b using a variational quantum algorithm
    with StronglyEntanglingLayers ansatz and Adam optimizer.

    Parameters
    ----------
    config : VQLSConfig | None
        Solver configuration. Uses defaults if None.
    """

    def __init__(self, config: VQLSConfig | None = None) -> None:
        self.config = config or VQLSConfig()

    def solve(  # noqa: N803
        self, a_matrix: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], bool]:
        """Solve Ax=b using VQLS.

        Parameters
        ----------
        a_matrix : NDArray[np.float64]
            Design matrix of shape (system_size, num_coefficients).
        b : NDArray[np.float64]
            Observation vector of shape (system_size,).

        Returns
        -------
        tuple[NDArray[np.float64], bool]
            (coefficients, converged) tuple. If not converged, caller
            should fall back to classical solver.
        """
        require_quantum()
        import pennylane as qml

        system_size = a_matrix.shape[0]
        num_coefficients = a_matrix.shape[1]
        n_qubits = calculate_n_qubits(system_size)
        full_size = 1 << n_qubits

        # Pad design matrix rows to power of 2 if needed
        a_padded = np.zeros((full_size, num_coefficients), dtype=np.float64)
        a_padded[:system_size, :] = a_matrix

        # Prepare b state and record original norm for denormalization
        b_norm = np.linalg.norm(b)
        b_state = prepare_b_state(b)

        # Build cost function
        dev = qml.device(self.config.device_name, wires=n_qubits)

        @qml.qnode(dev)
        def circuit(params: NDArray) -> NDArray[np.complex128]:
            """Ansatz circuit returning the statevector."""
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.state()

        def cost_fn(params: NDArray) -> float:
            """VQLS cost: 1 - |<b|A|psi>|^2 / (<psi|A^T A|psi> * <b|b>)."""
            state = circuit(params)
            psi = np.array(state, dtype=np.complex128)

            # Apply design matrix to statevector coefficients
            a_psi = a_padded @ psi[:num_coefficients]
            # Pad to full_size for inner product with b_state
            a_psi_full = np.zeros(full_size, dtype=np.complex128)
            a_psi_full[: len(a_psi)] = a_psi

            b_state_c = b_state.astype(np.complex128)
            overlap = np.abs(np.dot(np.conj(b_state_c), a_psi_full)) ** 2
            norm_a_psi = np.real(np.dot(np.conj(a_psi_full), a_psi_full))
            norm_b = np.real(np.dot(np.conj(b_state_c), b_state_c))

            denominator = norm_a_psi * norm_b
            if denominator < 1e-15:
                return 1.0

            return float(1.0 - overlap / denominator)

        # Initialize parameters
        param_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.config.n_layers, n_wires=n_qubits
        )
        params = np.random.default_rng(42).normal(0, 0.1, size=param_shape)

        # Optimize with Adam (manual implementation for numpy compatibility)
        converged = False
        best_params = params.copy()
        best_cost = float("inf")

        # Adam optimizer state
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = self.config.stepsize

        for step in range(self.config.max_iterations):
            current_cost = cost_fn(params)

            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()

            if current_cost < self.config.convergence_threshold:
                converged = True
                break

            # Compute gradient via finite differences
            grad = np.zeros_like(params)
            delta = 1e-4
            flat_params = params.flatten()
            for i in range(len(flat_params)):
                flat_plus = flat_params.copy()
                flat_plus[i] += delta
                flat_minus = flat_params.copy()
                flat_minus[i] -= delta
                grad.flat[i] = (
                    cost_fn(flat_plus.reshape(params.shape))
                    - cost_fn(flat_minus.reshape(params.shape))
                ) / (2 * delta)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

        # Extract coefficients from final statevector
        final_state = circuit(best_params)
        coefficients = extract_coefficients(
            np.array(final_state, dtype=np.complex128),
            num_coefficients,
            b_norm,
        )

        return coefficients, converged
