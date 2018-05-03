'''
Tests for det(K)
'''

from numpy.testing import assert_almost_equal
from unittest import TestCase

from . import test_two_bar_truss_model

class TestDetK(TestCase):
    def setUp(self):
        # two bar truss model
        self.model = test_two_bar_truss_model.get_model()

    def test_initial_model(self):
        model = self.model.get_duplicate()
        model.solve_det_k()
        actual_value = model.det_k
        expected_value = 0.4999999999999997
        assert_almost_equal(actual_value, expected_value)

    def test_after_solution(self):
        model = self.model.get_duplicate()
        model.lam = 0.1
        model.perform_non_linear_solution_step(strategy="load-control")
        model.solve_det_k()
        actual_value = model.det_k
        expected_value = 0.19510608810631772
        assert_almost_equal(actual_value, expected_value)
