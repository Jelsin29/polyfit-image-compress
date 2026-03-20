"""Tests for polynomial model definitions."""

from __future__ import annotations

import numpy as np
import pytest

from polyfit_compress.models import LinearModel, QuadraticModel, PolynomialModel


class TestLinearModel:
    def test_num_coefficients(self) -> None:
        model = LinearModel()
        assert model.num_coefficients == 3

    def test_name(self) -> None:
        model = LinearModel()
        assert model.name == "linear"

    def test_design_matrix_shape_8x8(self) -> None:
        model = LinearModel()
        A = model.build_design_matrix(8)
        assert A.shape == (64, 3)

    def test_design_matrix_shape_16x16(self) -> None:
        model = LinearModel()
        A = model.build_design_matrix(16)
        assert A.shape == (256, 3)

    def test_design_matrix_first_column_ones(self) -> None:
        model = LinearModel()
        A = model.build_design_matrix(8)
        np.testing.assert_array_equal(A[:, 0], np.ones(64))

    def test_invalid_block_size_zero(self) -> None:
        model = LinearModel()
        with pytest.raises(ValueError):
            model.build_design_matrix(0)

    def test_implements_protocol(self) -> None:
        assert isinstance(LinearModel(), PolynomialModel)


class TestQuadraticModel:
    def test_num_coefficients(self) -> None:
        model = QuadraticModel()
        assert model.num_coefficients == 6

    def test_name(self) -> None:
        model = QuadraticModel()
        assert model.name == "quadratic"

    def test_design_matrix_shape_8x8(self) -> None:
        model = QuadraticModel()
        A = model.build_design_matrix(8)
        assert A.shape == (64, 6)

    def test_design_matrix_shape_4x4(self) -> None:
        model = QuadraticModel()
        A = model.build_design_matrix(4)
        assert A.shape == (16, 6)

    def test_design_matrix_contains_cross_term(self) -> None:
        """Column 3 should be x*y."""
        model = QuadraticModel()
        A = model.build_design_matrix(4)
        # For a 4x4 block, check that column 3 = col1 * col2
        np.testing.assert_array_almost_equal(A[:, 3], A[:, 1] * A[:, 2])

    def test_design_matrix_contains_squared_terms(self) -> None:
        """Columns 4 and 5 should be x^2 and y^2."""
        model = QuadraticModel()
        A = model.build_design_matrix(4)
        np.testing.assert_array_almost_equal(A[:, 4], A[:, 1] ** 2)
        np.testing.assert_array_almost_equal(A[:, 5], A[:, 2] ** 2)

    def test_implements_protocol(self) -> None:
        assert isinstance(QuadraticModel(), PolynomialModel)
