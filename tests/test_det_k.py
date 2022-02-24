'''
Tests for det(K)
'''

import pytest
import nfem
from numpy.testing import assert_almost_equal


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=1, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


def test_det_k_initial_model(model):
    model.solve_det_k()
    actual_value = model.det_k
    expected_value = 0.4999999999999997
    assert_almost_equal(actual_value, expected_value)


def test_det_k_after_solution(model):
    model.load_factor = 0.1
    model.perform_load_control_step()
    model.solve_det_k()
    actual_value = model.det_k
    expected_value = 0.19510608810631772
    assert_almost_equal(actual_value, expected_value)
