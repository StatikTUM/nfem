'''
Tests for model creation
'''
from unittest import TestCase

from nfem import Model

class TestModel(TestCase):
    def setUp(self):
        self.model = Model('Two-Bar Truss')

        self.model.add_node(id='A', x=0, y=0, z=0)
        self.model.add_node(id='B', x=1, y=1, z=0)
        self.model.add_node(id='C', x=2, y=0, z=0)

        self.model.add_truss_element(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
        self.model.add_truss_element(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)

        self.model.add_single_load(id='load 1', node_id='B', fv=-1)

        self.model.add_dirichlet_condition(node_id='A', dof_types='uvw', value=0)
        self.model.add_dirichlet_condition(node_id='B', dof_types='w', value=0)
        self.model.add_dirichlet_condition(node_id='C', dof_types='uvw', value=0)

    def test_node_container(self):
        actual = len(self.model._nodes)
        expected = 3
        assert(actual == expected)

    def test_elements_container(self):
        actual = len(self.model.elements) 
        expected = 3
        assert(actual == expected)

    def test_dirichlet_container(self):
        actual = len(self.model.dirichlet_conditions)
        expected = 7
        assert(actual == expected)

def get_model():
    '''Make this model available for further tests'''
    test = TestModel()
    test.setUp()
    return test.model