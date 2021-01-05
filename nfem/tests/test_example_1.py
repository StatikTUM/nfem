import pytest
import nfem
from numpy.testing import assert_almost_equal


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node('A', x=0, y=0, z=0, support='xyz')
    model.add_node('B', x=1, y=1, z=0, support='z', fy=-1)
    model.add_node('C', x=2, y=0, z=0, support='xyz')

    model.add_truss('1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss('2', node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


@pytest.fixture
def load_curve():
    return [0.01, 0.02, 0.03, 0.05, 0.10, 0.136, 0.15, 0.30, 0.50, 0.70, 1.00]


def test_linear(model, load_curve):
    for load_factor in load_curve:
        model = model.get_duplicate()
        model.load_factor = load_factor
        model.perform_linear_solution_step()

    actual = model.load_displacement_curve(('B', 'v'), skip_iterations=False)

    assert_almost_equal(actual.T, [
        [0.0, 0.0],
        [-0.0141421356237309, 0.01],
        [-0.028284271247461912, 0.02],
        [-0.04242640687119281, 0.03],
        [-0.07071067811865483, 0.05],
        [-0.14142135623730956, 0.1],
        [-0.19233304448274102, 0.136],
        [-0.21213203435596428, 0.15],
        [-0.42426406871192857, 0.3],
        [-0.7071067811865477, 0.5],
        [-0.9899494936611667, 0.7],
        [-1.4142135623730954, 1.0],
    ])


def test_nonlinear(model, load_curve):
    for load_factor in load_curve:
        model = model.get_duplicate()
        model.predict_tangential(strategy='lambda', value=load_factor)
        model.perform_non_linear_solution_step(strategy='load-control')

    actual = model.load_displacement_curve(('B', 'v'), skip_iterations=False)

    assert_almost_equal(actual.T[-1], [-2.6480863731391198, 1.0])
