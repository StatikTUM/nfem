import nfem

from numpy.testing import assert_allclose

def test_urs():
    # desired force

    f1 = 5
    f2 = 6
    f3 = 5

    # classic FEM nodel

    model = nfem.Model()

    model.add_node('A', x=0, y=0, z=0, support='z')
    model.add_node('B', x=-1, y=1, z=0, support='xyz')
    model.add_node('C', x=1, y=0, z=0, support='xyz')
    model.add_node('D', x=0, y=2, z=0, support='xyz')

    model.add_truss('1', node_a='A', node_b='B', youngs_modulus=0, area=1, prestress=f1)
    model.add_truss('2', node_a='A', node_b='C', youngs_modulus=0, area=1, prestress=f2)
    model.add_truss('3', node_a='A', node_b='D', youngs_modulus=0, area=1, prestress=f3)

    model.elements['1'].updated_reference_strategy = True
    model.elements['2'].updated_reference_strategy = True
    model.elements['3'].updated_reference_strategy = True

    model.load_factor = 1;

    model.solve_load_control()

    assert_allclose(model.nodes["A"].location, [-0.125, 1.125, 0], atol=1e-4)
    assert_allclose(model.elements['1'].normal_force, f1)
    assert_allclose(model.elements['2'].normal_force, f2)
    assert_allclose(model.elements['3'].normal_force, f3)