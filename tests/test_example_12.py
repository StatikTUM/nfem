import pytest
import nfem
from numpy.testing import assert_almost_equal


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node('A', x=0, y=0, z=0, support='xyz')
    model.add_node('B', x=4, y=1, z=0, support='xz')

    model.add_truss('1', node_a='A', node_b='B', youngs_modulus=10, area=1)

    model.load_factor = 1

    return model


def test_a(model):
    model.nodes['B'].fy = 0.5

    model = model.get_duplicate()
    model.perform_load_control_step()

    assert_almost_equal(model.nodes['B'].v, 1.0875149418525858)


def test_b(model):
    model.nodes['B'].fy = 1
    model.add_spring('ks', node='B', ky=0.8)

    model = model.get_duplicate()
    model.perform_load_control_step()

    assert_almost_equal(model.nodes['B'].v, 0.8502106048504516)


def test_c(model):
    model.nodes['A'].fx = 0.5
    model.nodes['A'].support_x = False

    model.nodes['B'].fy = 1

    model.elements['1'].youngs_modulus = 20

    model.add_spring('ks', node='B', ky=0.8)

    model = model.get_duplicate()
    model.perform_load_control_step()

    assert_almost_equal(model.nodes['A'].u, 1.3553296964261339)
    assert_almost_equal(model.nodes['B'].v, 1.9462770095396955)
