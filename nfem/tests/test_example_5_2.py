import pytest
import nfem


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node(id='1', x=0, y=0, z=0, support='xyz')
    model.add_node(id='2', x=1, y=0, z=0, support='yz')
    model.add_node(id='3', x=2, y=0, z=0, support='xyz')
    model.add_node(id='4', x=0, y=1, z=0, support='xz')
    model.add_node(id='5', x=1, y=1, z=0, fz=-1)
    model.add_node(id='6', x=2, y=1, z=0, support='xz')
    model.add_node(id='7', x=0, y=2, z=0, support='xyz')
    model.add_node(id='8', x=1, y=2, z=0, support='yz')
    model.add_node(id='9', x=2, y=2, z=0, support='xyz')

    # outer elements
    model.add_truss(id=1, node_a='1', node_b='2', youngs_modulus=1, area=1)
    model.add_truss(id=2, node_a='2', node_b='3', youngs_modulus=1, area=1)
    model.add_truss(id=3, node_a='3', node_b='6', youngs_modulus=1, area=1)
    model.add_truss(id=4, node_a='6', node_b='9', youngs_modulus=1, area=1)
    model.add_truss(id=5, node_a='8', node_b='9', youngs_modulus=1, area=1)
    model.add_truss(id=6, node_a='7', node_b='8', youngs_modulus=1, area=1)
    model.add_truss(id=7, node_a='4', node_b='7', youngs_modulus=1, area=1)
    model.add_truss(id=8, node_a='1', node_b='4', youngs_modulus=1, area=1)

    # inner elements
    model.add_truss(id=9, node_a='2', node_b='5', youngs_modulus=1, area=1)
    model.add_truss(id=10, node_a='5', node_b='8', youngs_modulus=1, area=1)
    model.add_truss(id=11, node_a='4', node_b='5', youngs_modulus=1, area=1)
    model.add_truss(id=12, node_a='5', node_b='6', youngs_modulus=1, area=1)

    return model
