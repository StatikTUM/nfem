'''
Tests for the eigenvalue analysis
'''

import pytest
import nfem
from numpy.testing import assert_almost_equal


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=3, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


def test_branch_switching(model):
    bifurcation_model = model.get_duplicate()
    bifurcation_model.load_factor = 0.1
    bifurcation_model.perform_load_control_step()

    critical_model = nfem.bracketing(bifurcation_model)

    predicted_model = critical_model.get_duplicate()

    predicted_model.predict_tangential(strategy='arc-length')

    predicted_model.combine_prediction_with_eigenvector(beta=1.0)

    predicted_model.scale_prediction(factor=20000)

    predicted_model.perform_arc_length_control_step()

    # compare lambda
    actual = predicted_model.load_factor
    expected = 0.1673296703967696
    assert_almost_equal(actual, expected, decimal=4)

    # compare horizontal displacement
    actual = predicted_model.nodes['B'].u
    expected = 0.031161882052543888
    assert_almost_equal(actual, expected)
