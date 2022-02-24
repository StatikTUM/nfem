import pytest
import nfem
from numpy.testing import assert_almost_equal, assert_array_almost_equal


@pytest.fixture
def model_1():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=0, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='z', fy=-1)
    model.add_node(id='D', x=3, y=0, z=0, support='z', fy=-1)
    model.add_node(id='E', x=4, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)
    model.add_truss(id='3', node_a='C', node_b='D', youngs_modulus=1, area=1)
    model.add_truss(id='4', node_a='D', node_b='E', youngs_modulus=1, area=1)

    return model


@pytest.fixture
def model_2():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=0, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='z', fy=-1)
    model.add_node(id='D', x=3, y=0, z=0, support='z', fy=-1)
    model.add_node(id='E', x=4, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1, prestress=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1, prestress=1)
    model.add_truss(id='3', node_a='C', node_b='D', youngs_modulus=1, area=1, prestress=1)
    model.add_truss(id='4', node_a='D', node_b='E', youngs_modulus=1, area=1, prestress=1)

    return model


def test_linear_solution_fails(model_1):
    with pytest.raises(RuntimeError):
        model = model_1.get_duplicate()
        model.load_factor = 1.0
        model.perform_linear_solution_step()


def test_nonlinear_solution_fails(model_1):
    with pytest.raises(RuntimeError):
        model = model_1.get_duplicate()
        model.predict_load_factor(value=0.1)
        model.perform_load_control_step()


def test_nonlinear_with_prestress(model_2):
    model = model_2.get_duplicate()
    model.predict_tangential(strategy='lambda', value=0.01)
    model.perform_load_control_step()

    model = model.get_duplicate()
    for element in model.elements:
        element.prestress = 0

    model.perform_load_control_step()

    assert_almost_equal(model.load_displacement_curve(('C', 'v')).T, [
        [0.0, 0.0],
        [-0.019998500395204243, 0.01],
        [-0.5027762101486632, 0.01],
    ])

    for step in range(7):
        model = model.get_duplicate()
        model.predict_tangential(strategy='arc-length')
        model.perform_arc_length_control_step()

    assert_almost_equal(model.load_displacement_curve(('C', 'v')).T[-1], [-3.33696062209198, 2.7207022912488386])


def test_nonlinear_without_prestress(model_1):
    model = model_1.get_duplicate()
    model.predict_dof_state(dof=('C', 'v'), value=-0.2)
    model.perform_displacement_control_step(dof=('C', 'v'))

    for step in range(15):
        model = model.get_duplicate()
        model.predict_tangential(strategy='delta-dof',  dof=('C', 'v'), value=-0.2)
        model.perform_displacement_control_step(dof=('C', 'v'))

    assert_almost_equal(model.load_displacement_curve(('C', 'v')).T, [
        [0, 0.0],
        [-0.2, 0.0006258749552864138],
        [-0.4, 0.00502517767054208],
        [-0.6000000000000001, 0.017028343698052933],
        [-0.8, 0.04047117406708145],
        [-1.0, 0.07905232475409663],
        [-1.2, 0.1362597616503292],
        [-1.4, 0.2153089908177671],
        [-1.5999999999999999, 0.3192544581116804],
        [-1.7999999999999998, 0.4510682197015355],
        [-1.9999999999999998, 0.6137121212797986],
        [-2.1999999999999997, 0.8101805413028882],
        [-2.4, 1.0435183439235614],
        [-2.6, 1.3168232048485564],
        [-2.8000000000000003, 1.6332401300737045],
        [-3.0000000000000004, 1.9959531892454738],
        [-3.2000000000000006, 2.4081771325499424],
    ])


def test_stiffness_without_prestress(model_1):
    model = model_1.get_duplicate()

    print(model.get_stiffness())

    assert_array_almost_equal(model.get_stiffness(),
        [[ 2, 0, -1, 0,  0, 0],
         [ 0, 0,  0, 0,  0, 0],
         [-1, 0,  2, 0, -1, 0],
         [ 0, 0,  0, 0,  0, 0],
         [ 0, 0, -1, 0,  2, 0],
         [ 0, 0,  0, 0,  0, 0]])


def test_stiffness_with_prestress(model_2):
    model = model_2.get_duplicate()

    print(model.get_stiffness())

    assert_array_almost_equal(model.get_stiffness(),
        [[ 4,  0, -2,  0,  0,  0],
         [ 0,  2,  0, -1,  0,  0],
         [-2,  0,  4,  0, -2,  0],
         [ 0, -1,  0,  2,  0, -1],
         [ 0,  0, -2,  0,  4,  0],
         [ 0,  0,  0, -1,  0,  2]])
