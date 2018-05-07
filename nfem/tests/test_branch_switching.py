'''
Tests for the eigenvalue analysis
'''

from numpy.testing import assert_almost_equal
from unittest import TestCase

from . import test_two_bar_truss_model
from ..bracketing import bracketing
from .. import *

class TestBranchSwitching(TestCase):
    def setUp(self):
        # two bar truss model
        self.model = test_two_bar_truss_model.get_model()
        # modify two bar truss so it has a bifurcation point
        self.model._nodes['B'].reference_y = 3.0
        self.model._nodes['B'].y = 3.0

    def test_branch_switching(self):
        bifurcation_model = self.model.get_duplicate()
        bifurcation_model.lam = 0.1
        bifurcation_model.perform_non_linear_solution_step(strategy='load-control')

        critical_model = bracketing(bifurcation_model)

        predicted_model = critical_model.get_duplicate()

        predicted_model.predict_tangential(strategy='arc-length')

        predicted_model.combine_prediction_with_eigenvector(factor=1.0)

        #predicted_model.scale_prediction(factor=2.0)

        predicted_model.perform_non_linear_solution_step(strategy='arc-length-control')

        # compare lambda
        actual = predicted_model.lam
        expected = 0.16726702267341678
        assert_almost_equal(actual, expected)

        # compare horizontal displacement
        actual = predicted_model.get_dof_state(dof=('B','u'))
        assert(actual!=0.0)

        # animated plot
        animation = Animation3D()
        animation.show(predicted_model)

if __name__ == "__main__":
    test = TestBranchSwitching()
    test.setUp()
    test.test_branch_switching()