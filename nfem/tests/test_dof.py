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


def test_dof_increment(dof):
    dof += 1.5

    assert_equal(dof.reference_value, 5)
    assert_equal(dof.value, 6.5)
    assert_equal(dof.delta, 1.5)


def test_dof_decrement(dof):
    dof -= 0.5

    assert_equal(dof.reference_value, 5)
    assert_equal(dof.value, 4.5)
    assert_equal(dof.delta, -0.5)


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
