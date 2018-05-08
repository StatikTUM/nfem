'''
Tests for the eigenvalue analysis
'''

from numpy.testing import assert_almost_equal
from unittest import TestCase

from . import test_two_bar_truss_model

class TestEigenvalueAnalysis(TestCase):
    def setUp(self):
        # two bar truss model
        self.model = test_two_bar_truss_model.get_model()

    def test_limit_point(self):
        limit_model = self.model.get_duplicate()
        limit_model.lam = 0.1
        limit_model.perform_non_linear_solution_step(strategy="load-control")
        limit_model.solve_eigenvalues()

        ev_actual = limit_model.first_eigenvalue
        ev_expected = 3.6959287916726304
        assert_almost_equal(ev_actual, ev_expected)
        # TODO test eigenvector [0.0, 1.0]

    def test_bifurcation_point(self):
        bifurcation_model = self.model.get_duplicate()
        bifurcation_model._nodes['B'].reference_y = 3.0
        bifurcation_model._nodes['B'].y = 3.0
        bifurcation_model.lam = 0.1
        bifurcation_model.perform_non_linear_solution_step(strategy="load-control")
        bifurcation_model.solve_eigenvalues()

        ev_actual = bifurcation_model.first_eigenvalue
        ev_expected = 1.7745968576086002
        assert_almost_equal(ev_actual, ev_expected)
        # TODO test eigenvector [1.0, 0.0]