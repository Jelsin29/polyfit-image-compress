"""Tests for VQLS solver and quantum utils."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_quantum.conftest import quantum_required


@quantum_required
class TestNormalizeState:
    """Tests for normalize_state utility."""

    def test_normalize_state_unit_vector(self) -> None:
        """Normalized vector has L2 norm approximately 1."""
        from polyfit_compress.quantum.utils import normalize_state

        vector = np.array([3.0, 4.0, 0.0])
        result = normalize_state(vector)
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_normalize_state_zero_vector(self) -> None:
        """Zero vector returns zeros."""
        from polyfit_compress.quantum.utils import normalize_state

        vector = np.zeros(5)
        result = normalize_state(vector)
        np.testing.assert_array_equal(result, np.zeros(5))


@quantum_required
class TestPrepareBState:
    """Tests for prepare_b_state utility."""

    def test_prepare_b_state_power_of_two_padding(self) -> None:
        """Output length is a power of 2."""
        from polyfit_compress.quantum.utils import prepare_b_state

        b = np.array([1.0, 2.0, 3.0])  # size 3 -> padded to 4
        result = prepare_b_state(b)
        assert len(result) == 4
        assert (len(result) & (len(result) - 1)) == 0  # power of 2 check

    def test_prepare_b_state_normalized(self) -> None:
        """Output is unit normalized."""
        from polyfit_compress.quantum.utils import prepare_b_state

        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = prepare_b_state(b)
        assert np.isclose(np.linalg.norm(result), 1.0)


@quantum_required
class TestCalculateNQubits:
    """Tests for calculate_n_qubits utility."""

    @pytest.mark.parametrize(
        ("system_size", "expected"),
        [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (16, 4),
            (64, 6),
        ],
    )
    def test_calculate_n_qubits(self, system_size: int, expected: int) -> None:
        """Various sizes map to correct qubit counts."""
        from polyfit_compress.quantum.utils import calculate_n_qubits

        assert calculate_n_qubits(system_size) == expected


@quantum_required
class TestExtractCoefficients:
    """Tests for extract_coefficients utility."""

    def test_extract_coefficients(self) -> None:
        """Extracts first N elements scaled by norm."""
        from polyfit_compress.quantum.utils import extract_coefficients

        statevector = np.array([0.5 + 0j, 0.3 + 0j, 0.1 + 0j, 0.0 + 0j])
        norm = 10.0
        result = extract_coefficients(statevector, num_coefficients=3, norm=norm)
        expected = np.array([5.0, 3.0, 1.0])
        np.testing.assert_allclose(result, expected)


@quantum_required
class TestVQLSConfig:
    """Tests for VQLSConfig defaults."""

    def test_vqls_config_defaults(self) -> None:
        """Default values are correct."""
        from polyfit_compress.quantum.vqls import VQLSConfig

        config = VQLSConfig()
        assert config.n_layers == 4
        assert config.max_iterations == 200
        assert config.stepsize == 0.1
        assert config.convergence_threshold == 1e-4
        assert config.device_name == "default.qubit"


@quantum_required
class TestVQLSSolver:
    """Tests for VQLSSolver."""

    def test_vqls_solver_small_system(self) -> None:
        """Solve a tiny linear system and compare to classical lstsq."""
        from polyfit_compress.models import LinearModel
        from polyfit_compress.quantum.vqls import VQLSConfig, VQLSSolver

        model = LinearModel()
        block_size = 2
        a_matrix = model.build_design_matrix(block_size)  # (4, 3)

        # Create a simple block of pixel values
        rng = np.random.default_rng(123)
        block = rng.uniform(50, 200, size=(block_size, block_size))
        b = block.flatten().astype(np.float64)

        # Classical solution
        classical_coeffs, _, _, _ = np.linalg.lstsq(a_matrix, b, rcond=None)

        # Quantum solution with relaxed convergence for speed
        config = VQLSConfig(
            n_layers=4,
            max_iterations=100,
            stepsize=0.1,
            convergence_threshold=1e-3,
        )
        solver = VQLSSolver(config=config)
        quantum_coeffs, converged = solver.solve(a_matrix, b)

        # VQLS returns coefficients on a normalized statevector, so they are
        # proportional to the classical solution (same direction, different scale).
        # Verify directional similarity via cosine similarity.
        cos_sim = np.dot(quantum_coeffs, classical_coeffs) / (
            np.linalg.norm(quantum_coeffs) * np.linalg.norm(classical_coeffs)
        )
        assert cos_sim > 0.95, f"Cosine similarity too low: {cos_sim}"

    def test_vqls_solver_returns_converged_flag(self) -> None:
        """Verify (coeffs, converged) tuple is returned."""
        from polyfit_compress.models import LinearModel
        from polyfit_compress.quantum.vqls import VQLSConfig, VQLSSolver

        model = LinearModel()
        a_matrix = model.build_design_matrix(2)  # (4, 3)
        b = np.array([100.0, 120.0, 130.0, 150.0])

        config = VQLSConfig(max_iterations=2)
        solver = VQLSSolver(config=config)
        result = solver.solve(a_matrix, b)

        assert isinstance(result, tuple)
        assert len(result) == 2
        coeffs, converged = result
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(converged, bool)
