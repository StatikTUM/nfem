import pytest
from nfem.dof import Dof
from copy import deepcopy
from numpy.testing import assert_equal


@pytest.fixture
def dof():
    return Dof('test', 5)


def test_dof_init(dof):
    assert_equal(dof.ref_value, 5)
    assert_equal(dof.value, 5)
    assert_equal(dof.delta, 0)


def test_dof_equality(dof):
    dof_2 = deepcopy(dof)
    dof_3 = Dof('test2', 5)

    assert(dof == dof_2)
    assert(dof == 'test')
    assert(dof != dof_3)
    assert(dof != 'test2')


def test_dof_ref_value(dof):
    dof.ref_value = 3

    assert_equal(dof.ref_value, 3)
    assert_equal(dof.value, 5)
    assert_equal(dof.delta, 2)


def test_dof_value(dof):
    dof.value = 3

    assert_equal(dof.ref_value, 5)
    assert_equal(dof.value, 3)
    assert_equal(dof.delta, -2)


def test_dof_delta(dof):
    dof.delta = 3

    assert_equal(dof.ref_value, 5)
    assert_equal(dof.value, 8)
    assert_equal(dof.delta, 3)


def test_dof_external_force(dof):
    dof.external_force = 3

    assert_equal(dof.external_force, 3)
