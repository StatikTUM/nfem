'''
Tests for the eigenvalue analysis
'''

import pytest
import nfem
from numpy.testing import assert_almost_equal


@pytest.fixture
def model_1():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=1, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


@pytest.fixture
def model_2():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=3, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


def test_bracketing_limit_point(model_1):
    model = model_1.get_duplicate()

    model.predict_tangential(strategy="lambda", value=0.05)
    model.perform_load_control_step()

    critical_model = nfem.bracketing(model)

    actual = critical_model.load_factor
    expected = 0.13607744543608463
    assert_almost_equal(actual, expected)


def test_bracketing_bifurcation_point(model_2):
    model = model_2.get_duplicate()

    model.predict_tangential(strategy="lambda", value=0.05)
    model.perform_load_control_step()

    critical_model = nfem.bracketing(model)
    actual = critical_model.load_factor
    expected = 0.16733018783531955
    assert_almost_equal(actual, expected)
