import nfem
import pytest
from numpy.testing import assert_equal


@pytest.fixture
def node():
    return nfem.Node('A', 4, 5, 6)


def test_node_init(node):
    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [4, 5, 6])
    assert_equal(node.displacement, [0, 0, 0])


def test_node_dof(node):
    assert_equal(node.dof('u'), node._dof_x)
    assert_equal(node.dof('v'), node._dof_y)
    assert_equal(node.dof('w'), node._dof_z)

    with pytest.raises(AttributeError):
        node.dof('a')


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


def test_node_fx(node):
    node.fx = 9

    assert_equal(node.fx, 9)
    assert_equal(node.external_force, [9, 0, 0])


def test_node_fy(node):
    node.fy = 9

    assert_equal(node.fy, 9)
    assert_equal(node.external_force, [0, 9, 0])


def test_node_fz(node):
    node.fz = 9

    assert_equal(node.fz, 9)
    assert_equal(node.external_force, [0, 0, 9])


def test_node_external_force(node):
    node.external_force = [1, 2, 3]

    assert_equal(node.external_force, [1, 2, 3])


def test_node_reference_location(node):
    node.reference_location = [3, 2, 1]

    assert_equal(node.reference_location, [3, 2, 1])
    assert_equal(node.location, [4, 5, 6])
    assert_equal(node.displacement, [1, 3, 5])


def test_node_location(node):
    node.location = [3, 2, 1]

    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [3, 2, 1])
    assert_equal(node.displacement, [-1, -3, -5])


def test_node_displacement(node):
    node.displacement = [3, 2, 1]

    assert_equal(node.reference_location, [4, 5, 6])
    assert_equal(node.location, [7, 7, 7])
    assert_equal(node.displacement, [3, 2, 1])


def test_node_support(node):
    assert_equal(node.support_x, False)
    assert_equal(node.support_y, False)
    assert_equal(node.support_z, False)

    node.support_x = True
    assert_equal(node.support_x, True)

    node.support_y = True
    assert_equal(node.support_y, True)

    node.support_z = True
    assert_equal(node.support_z, True)
