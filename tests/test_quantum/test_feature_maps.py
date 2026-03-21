"""Tests for quantum feature maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_quantum.conftest import quantum_required

if TYPE_CHECKING:
    from polyfit_compress.quantum.feature_maps import QuantumFeatureMap


@quantum_required
class TestQuantumFeatureMap:
    """Tests for QuantumFeatureMap class."""

    def _make_feature_map(self, n_features: int = 6) -> QuantumFeatureMap:
        from polyfit_compress.quantum.feature_maps import QuantumFeatureMap

        return QuantumFeatureMap(n_features=n_features)

    def test_feature_map_design_matrix_shape(self) -> None:
        """Design matrix has shape (block_size², n_features)."""
        n_features = 4
        block_size = 4
        fm = self._make_feature_map(n_features=n_features)
        matrix = fm.build_design_matrix(block_size)
        assert matrix.shape == (block_size * block_size, n_features)

    def test_feature_map_num_coefficients(self) -> None:
        """num_coefficients matches n_features."""
        fm = self._make_feature_map(n_features=5)
        assert fm.num_coefficients == 5

    def test_feature_map_values_bounded(self) -> None:
        """Expectation values of PauliZ are in [-1, 1]."""
        fm = self._make_feature_map(n_features=3)
        matrix = fm.build_design_matrix(block_size=2)
        assert np.all(matrix >= -1.0 - 1e-10)
        assert np.all(matrix <= 1.0 + 1e-10)

    def test_feature_map_deterministic(self) -> None:
        """Same inputs produce the same design matrix."""
        fm = self._make_feature_map(n_features=3)
        m1 = fm.build_design_matrix(block_size=2)
        m2 = fm.build_design_matrix(block_size=2)
        np.testing.assert_array_equal(m1, m2)

    @pytest.mark.parametrize("block_size", [2, 4])
    def test_feature_map_different_block_sizes(self, block_size: int) -> None:
        """Works correctly for different block sizes."""
        n_features = 3
        fm = self._make_feature_map(n_features=n_features)
        matrix = fm.build_design_matrix(block_size)
        assert matrix.shape == (block_size * block_size, n_features)
        assert matrix.dtype == np.float64
