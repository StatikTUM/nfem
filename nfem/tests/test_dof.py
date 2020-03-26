import pytest
from numpy.testing import assert_equal
from nfem.dof import Dof


@pytest.fixture
def dof():
    return Dof(5)


def test_dof_init(dof):
    assert_equal(dof.reference_value, 5)
    assert_equal(dof.value, 5)
    assert_equal(dof.delta, 0)


def test_dof_reference_value(dof):
    dof.reference_value = 3

    assert_equal(dof.reference_value, 3)
    assert_equal(dof.value, 5)
    assert_equal(dof.delta, 2)


def test_dof_value(dof):
    dof.value = 3

    assert_equal(dof.reference_value, 5)
    assert_equal(dof.value, 3)
    assert_equal(dof.delta, -2)


def test_dof_delta(dof):
    dof.delta = 3

    assert_equal(dof.reference_value, 5)
    assert_equal(dof.value, 8)
    assert_equal(dof.delta, 3)


def test_dof_external_force(dof):
    dof.external_force = 3

    assert_equal(dof.external_force, 3)

