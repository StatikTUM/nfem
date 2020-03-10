import pytest
from numpy.testing import assert_equal, assert_raises
from nfem.node import Node


@pytest.fixture
def node():
    return Node('A', 4, 5, 6)


@pytest.fixture
def node_with_displacement():
    node = Node('A', 4, 5, 6)
    node.u = 1
    node.v = 2
    node.w = 3
    return node


def test_node_get_dof_state(node_with_displacement):
    assert_equal(node_with_displacement.get_dof_state('u'), 1)
    assert_equal(node_with_displacement.get_dof_state('v'), 2)
    assert_equal(node_with_displacement.get_dof_state('w'), 3)
    with pytest.raises(AttributeError):
        node_with_displacement.get_dof_state('invalid')


def test_node_set_dof_state(node):
    node.set_dof_state('u', 1)
    node.set_dof_state('v', 2)
    node.set_dof_state('w', 3)
    assert_equal(node.get_dof_state('u'), 1)
    assert_equal(node.get_dof_state('v'), 2)
    assert_equal(node.get_dof_state('w'), 3)
    with pytest.raises(AttributeError):
        node.set_dof_state('invalid', 0)


def test_node_init(node):
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [4, 5, 6])
    assert_equal(node.displacement, [0, 0, 0])


def test_node_reference_x(node):
    node.reference_x = 9

    assert_equal(node.reference_x, 9)
    assert_equal(node.reference_location, [9, 5, 6])
    assert_equal(node.location, [4, 5, 6])


def test_node_reference_y(node):
    node.reference_y = 9

    assert_equal(node.reference_y, 9)
    assert_equal(node.reference_location, [4, 9, 6])
    assert_equal(node.location, [4, 5, 6])


def test_node_reference_z(node):
    node.reference_z = 9

    assert_equal(node.reference_z, 9)
    assert_equal(node.reference_location, [4, 5, 9])
    assert_equal(node.location, [4, 5, 6])


def test_node_x(node):
    node.x = 9

    assert_equal(node.x, 9)
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [9, 5, 6])


def test_node_y(node):
    node.y = 9

    assert_equal(node.y, 9)
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [4, 9, 6])


def test_node_z(node):
    node.z = 9

    assert_equal(node.z, 9)
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [4, 5, 9])


def test_node_u(node):
    node.u = 9

    assert_equal(node.u, 9)
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [13, 5, 6])
    assert_equal(node.displacement, [9, 0, 0])


def test_node_v(node):
    node.v = 9

    assert_equal(node.v, 9)
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [4, 14, 6])
    assert_equal(node.displacement, [0, 9, 0])


def test_node_w(node):
    node.w = 9

    assert_equal(node.w, 9)
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [4, 5, 15])
    assert_equal(node.displacement, [0, 0, 9])
