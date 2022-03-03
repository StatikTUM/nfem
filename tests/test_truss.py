import pytest

import nfem

from numpy.testing import assert_equal, assert_almost_equal


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)


@pytest.fixture
def undeformed_truss():
    node_a = nfem.Node('A', 1, 2, 3)
    node_b = nfem.Node('B', 4, 6, 3)
    return nfem.Truss('1', node_a, node_b, 2, 1)


@pytest.fixture
def truss():
    node_a = nfem.Node('A', 1, 2, 3)
    node_b = nfem.Node('B', 4, 6, 3)
    node_b.u = 3
    node_b.v = 4
    node_b.w = 0
    return nfem.Truss('1', node_a, node_b, 2, 1)


@pytest.fixture
def truss_xls():
    node_a = nfem.Node('A', 0, 0, 0)
    node_b = nfem.Node('B', 1, 2, 3)
    node_b.u += 0.1
    return nfem.Truss('1', node_a, node_b, 1, 1)


def test_truss_ref_length(truss):
    assert_equal(truss.ref_length, 5)


def test_truss_act_length(truss):
    assert_equal(truss.length, 10)


def test_truss_linear_strain_is_zero(undeformed_truss):
    assert_equal(undeformed_truss.compute_epsilon_lin(), 0)


def test_truss_green_lagrange_strain_is_zero(undeformed_truss):
    assert_equal(undeformed_truss.compute_epsilon_gl(), 0)


def test_truss_engineering_strain(truss_xls):
    assert_almost_equal(truss_xls.compute_epsilon_lin(), 0.0071428571428572)


def test_truss_green_lagrange_strain(truss_xls):
    assert_almost_equal(truss_xls.compute_epsilon_gl(), 0.0075)


def test_truss_normal_force(truss_xls):
    assert_almost_equal(truss_xls.normal_force, 0.007556040629853737)


def test_truss_compute_linear_r(truss_xls):
    r_act = truss_xls.compute_linear_r()

    r_exp = [-0.001909, -0.003818, -0.005727, +0.001909, +0.003818, +0.005727]

    assert_almost_equal(r_act, r_exp)


def test_truss_compute_linear_r_prestessed(undeformed_truss):
    undeformed_truss.youngs_modulus = 3
    undeformed_truss.prestress = 5

    r_act = undeformed_truss.compute_linear_r()

    assert_almost_equal(r_act, [-3, -4, 0, 3, 4, 0])


def test_truss_compute_linear_r_prestress_initial_displacement(undeformed_truss):
    undeformed_truss.youngs_modulus = 3
    undeformed_truss.node_b.v = -4

    r_act = undeformed_truss.compute_linear_r()

    assert_almost_equal(r_act, [1.152, 1.536, 0, -1.152, -1.536, 0])


def test_truss_compute_linear_r_initial_displacement(undeformed_truss):
    undeformed_truss.youngs_modulus = 3
    undeformed_truss.prestress = 5
    undeformed_truss.node_b.v = -4

    r_act = undeformed_truss.compute_linear_r()

    assert_almost_equal(r_act, [-1.848, -2.464, 0, 1.848, 2.464, 0])

    truss.prestress = 0


def test_truss_compute_linear_k(truss_xls):
    k_act = truss_xls.compute_linear_k()

    # values from Truss.xls
    k_exp = [
        [+0.019090089, +0.038180177, +0.057270266, -0.019090089, -0.038180177, -0.057270266],
        [+0.038180177, +0.076360355, +0.114540532, -0.038180177, -0.076360355, -0.114540532],
        [+0.057270266, +0.114540532, +0.171810798, -0.057270266, -0.114540532, -0.171810798],
        [-0.019090089, -0.038180177, -0.057270266, +0.019090089, +0.038180177, +0.057270266],
        [-0.038180177, -0.076360355, -0.114540532, +0.038180177, +0.076360355, +0.114540532],
        [-0.057270266, -0.114540532, -0.171810798, +0.057270266, +0.114540532, +0.171810798],
    ]

    assert_almost_equal(k_act, k_exp)


def test_truss_solve_linear_prestessed(undeformed_truss):
    undeformed_truss.youngs_modulus = 3

    # positive prestress

    undeformed_truss.prestress = 5

    r = undeformed_truss.compute_linear_r()
    k = undeformed_truss.compute_linear_k()

    d = -r[4] / k[4, 4]

    assert_almost_equal(d, -10.41666666666666)

    # negative prestress

    undeformed_truss.prestress = -5

    r = undeformed_truss.compute_linear_r()
    k = undeformed_truss.compute_linear_k()

    d = -r[4] / k[4, 4]

    assert_almost_equal(d, 10.41666666666666)


def test_truss_compute_r(truss_xls):
    r_act = truss_xls.compute_r()

    r_exp = [-0.0022049, -0.0040089, -0.0060134, +0.0022049, +0.0040089, 0.0060134]

    assert_almost_equal(r_act, r_exp)


def test_truss_compute_k(truss_xls):
    k_act = truss_xls.compute_k()

    # values from Truss.xls
    k_exp = [
        [+0.025103466943026, +0.041998194741606, +0.062997292612409, -0.025103466943026, -0.041998194741606, -0.062997292612409],
        [+0.041998194741606, +0.078364814314340, +0.114540532000000, -0.041998194741606, -0.078364814314340, -0.114540532000000],
        [+0.062997292612409, +0.114540532000000, +0.173815257314340, -0.062997292612409, -0.114540532000000, -0.173815257314340],
        [-0.025103466943026, -0.041998194741606, -0.062997292612409, +0.025103466943026, +0.041998194741606, +0.062997292612409],
        [-0.041998194741606, -0.078364814314340, -0.114540532000000, +0.041998194741606, +0.078364814314340, +0.114540532000000],
        [-0.062997292612409, -0.114540532000000, -0.173815257314340, +0.062997292612409, +0.114540532000000, +0.173815257314340],
    ]

    assert_almost_equal(k_act, k_exp)


def test_truss_compute_km(truss_xls):
    k_act = truss_xls.compute_km()

    k_exp = [
        [+0.023099007336717, +0.041998195157667, +0.062997292736500, -0.023099007336717, -0.041998195157667, -0.062997292736500],
        [+0.041998195157667, +0.076360354832121, +0.114540532248182, -0.041998195157667, -0.076360354832121, -0.114540532248182],
        [+0.062997292736500, +0.114540532248182, +0.171810798372273, -0.062997292736500, -0.114540532248182, -0.171810798372273],
        [-0.023099007336717, -0.041998195157667, -0.062997292736500, +0.023099007336717, +0.041998195157667, +0.062997292736500],
        [-0.041998195157667, -0.076360354832121, -0.114540532248182, +0.041998195157667, +0.076360354832121, +0.114540532248182],
        [-0.062997292736500, -0.114540532248182, -0.171810798372273, +0.062997292736500, +0.114540532248182, +0.171810798372273],
    ]

    assert_almost_equal(k_act, k_exp)


def test_truss_compute_kd(truss_xls):
    k_act = truss_xls.compute_kd()

    # values from Truss.xls
    k_exp = [
        [+0.004008918628686, +0.003818017741606, +0.005727026612409, -0.004008918628686, -0.003818017741606, -0.005727026612409],
        [+0.003818017741606, +0.000000000000000, +0.000000000000000, -0.003818017741606, +0.000000000000000, +0.000000000000000],
        [+0.005727026612409, +0.000000000000000, +0.000000000000000, -0.005727026612409, +0.000000000000000, +0.000000000000000],
        [-0.004008918628686, -0.003818017741606, -0.005727026612409, +0.004008918628686, +0.003818017741606, +0.005727026612409],
        [-0.003818017741606, +0.000000000000000, +0.000000000000000, +0.003818017741606, +0.000000000000000, +0.000000000000000],
        [-0.005727026612409, +0.000000000000000, +0.000000000000000, +0.005727026612409, +0.000000000000000, +0.000000000000000],
    ]

    assert_almost_equal(k_act, k_exp)


def test_truss_compute_kg(truss_xls):
    k_act = truss_xls.compute_kg()

    # values from Truss.xls
    k_exp = [
        [+2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00, -2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00],
        [+0.00000000000E+00, +2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00, -2.00445931434E-03, +0.00000000000E+00],
        [+0.00000000000E+00, +0.00000000000E+00, +2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00, -2.00445931434E-03],
        [-2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00, +2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00],
        [+0.00000000000E+00, -2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00, +2.00445931434E-03, +0.00000000000E+00],
        [+0.00000000000E+00, +0.00000000000E+00, -2.00445931434E-03, +0.00000000000E+00, +0.00000000000E+00, +2.00445931434E-03]
    ]

    assert_almost_equal(k_act, k_exp)
