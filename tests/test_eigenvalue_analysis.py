'''
Tests for the eigenvalue analysis
'''

import nfem
import pytest
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


def test_limit_point(model_1):
    model = model_1
    model.load_factor = 0.1
    model.perform_load_control_step()
    model.solve_eigenvalues()

    ev_actual = model.first_eigenvalue
    ev_expected = 3.6959287916726304  # safety_factor
    assert_almost_equal(ev_actual, ev_expected)

    # test eigenvector [0.0, 1.0]
    eigenvector_model = model.first_eigenvector_model
    u_actual = eigenvector_model.nodes['B'].u
    u_expected = 0.0
    assert_almost_equal(u_actual, u_expected)
    v_actual = eigenvector_model.nodes['B'].v
    v_expected = 1.0
    assert_almost_equal(v_actual, v_expected)


def test_bifurcation_point(model_2):
    model = model_2
    model.load_factor = 0.1
    model.perform_load_control_step()
    model.solve_eigenvalues()

    ev_actual = model.first_eigenvalue
    ev_expected = 1.7745968576086002  # safety_factor
    assert_almost_equal(ev_actual, ev_expected)

    # test eigenvector [1.0, 0.0]
    eigenvector_model = model.first_eigenvector_model
    u_actual = eigenvector_model.nodes['B'].u
    u_expected = 1.0
    assert_almost_equal(u_actual, u_expected)
    v_actual = eigenvector_model.nodes['B'].v
    v_expected = 0.0
    assert_almost_equal(v_actual, v_expected)


def test_lpb_limit_point(model_1):
    model = model_1
    model.load_factor = 0.1
    model.perform_linear_solution_step()
    model.solve_linear_eigenvalues()

    lam_crit_actual = model.first_eigenvalue * model.load_factor
    lam_crit_expected = 0.7071067811865452
    assert_almost_equal(lam_crit_actual, lam_crit_expected)

    # FIXME eigenvector shows bifurcation instead of limit for h>=1.0...
