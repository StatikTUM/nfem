'''
Tests for model creation
'''
import pytest
import numpy
from numpy.testing import assert_almost_equal, assert_equal

import nfem


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=1, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


def test_model_dofs(model):
    assert_equal(len(model.dofs), 2)


def test_model_nodes(model):
    assert_equal(len(model.nodes), 3)


def test_model_elements(model):
    assert_equal(len(model.elements), 2)


def test_initial_zero_dof_increment(model):
    assert_equal(model.get_dof_increment(dof=('B', 'v')), 0)


def test_initial_zero_lam_increment(model):
    assert_equal(model.get_lam_increment(), 0)


def test_initial_zero_increment_vector(model):
    assert_equal(model.get_increment_vector(), [0, 0, 0])


def test_initial_zero_increment_norm(model):
    assert_equal(model.get_increment_norm(), 0)


def test_integer_node_key_raises(model):
    with pytest.raises(TypeError):
        model.add_node(id=9, x=0, y=0, z=0)


def test_duplicate_node_key_raises(model):
    with pytest.raises(KeyError):
        model.add_node(id='A', x=0, y=0, z=0)


def test_integer_truss_key_raises(model):
    with pytest.raises(TypeError):
        model.add_truss(id=9, node_a='A', node_b='B', youngs_modulus=1, area=1)

    with pytest.raises(TypeError):
        model.add_truss(id='9', node_a=0, node_b='B', youngs_modulus=1, area=1)

    with pytest.raises(TypeError):
        model.add_truss(id='9', node_a='A', node_b=1, youngs_modulus=1, area=1)


def test_duplicate_truss_key_raises(model):
    with pytest.raises(KeyError):
        model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)


def test_invalid_node_for_truss_raises(model):
    with pytest.raises(KeyError):
        model.add_truss(id='9', node_a='Z', node_b='B', youngs_modulus=1, area=1)

    with pytest.raises(KeyError):
        model.add_truss(id='9', node_a='A', node_b='Z', youngs_modulus=1, area=1)


def test_get_duplicate(model):
    model_1 = model.get_duplicate()
    assert_equal(model_1.get_previous_model(), model)

    model_2 = model_1.get_duplicate(branch=True)
    assert_equal(model_2.get_previous_model(), model)


def test_new_timestep(model):
    new_timestep = model.new_timestep()
    assert_equal(new_timestep.get_previous_model(), model)


def test_predict_load_increment(model):
    model.predict_load_increment(5)


def test_predict_dof_state(model):
    model.predict_dof_state(dof=('B', 'v'), value=5)


def test_predict_dof_increment(model):
    model.predict_dof_increment(dof=('B', 'v'), value=5)


def test_predict_with_last_increment(model):
    with pytest.raises(AttributeError):
        model.predict_with_last_increment(5)


def test_add_element():
    class Element:
        def __init__(self, id, nodes, k):
            self.id = id
            self.nodes = nodes
            self.k = k

        @property
        def dofs(self):
            dofs = []
            for node in self.nodes:
                dofs.append(node._dof_x)
                dofs.append(node._dof_y)
                dofs.append(node._dof_z)
            return dofs

        def compute_r(self):
            u = numpy.array([dof.delta for dof in self.dofs])
            return self.compute_k() @ u

        def compute_k(self):
            return numpy.eye(6) * self.k

    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=1, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_element(Element, id='1', nodes=['A', 'B'], k=5)
    model.add_element(Element, id='2', nodes=['B', 'C'], k=5)

    model.load_factor = 1
    model.perform_load_control_step()

    assert_almost_equal(model.nodes['B'].u, 0)
    assert_almost_equal(model.nodes['B'].v, -0.1)


def test_dof(model):
    assert_equal(model.dof('B.u').value, 1)
    assert_equal(model.dof('B.v').value, 1)
    assert_equal(model.dof('B.w').value, 0)
