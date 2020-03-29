'''
Tests for model creation
'''
import pytest
from numpy.testing import assert_equal

from nfem import Model


@pytest.fixture
def model():
    model = Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=1, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


def test_model_nodes(model):
    assert_equal(len(model.nodes), 3)


def test_model_elements(model):
    assert_equal(len(model.elements), 2)
