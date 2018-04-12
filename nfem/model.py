"""FIXME"""

from copy import deepcopy

import numpy as np
import numpy.linalg as la

from .assembler import Assembler

class ElementBase(object):
    """FIXME"""

    def Dofs(self):
        """FIXME"""
        raise NotImplementedError

    def Calculate(self, u, lam):
        """FIXME"""
        raise NotImplementedError

class Node(object):
    """FIXME"""

    def __init__(self, id, x, y, z):
        """FIXME"""

        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.reference_x = x
        self.reference_y = y
        self.reference_z = z

    def Update(self, dof_type, value):
        """FIXME"""

        if dof_type == 'u':
            self.x += value
            return

        if dof_type == 'v':
            self.y += value
            return

        if dof_type == 'w':
            self.z += value
            return

        raise RuntimeError('Node has no Dof of type {}'.format(dof_type))

class Truss(ElementBase):
    """FIXME"""

    def __init__(self, id, node_a, node_b, youngs_modulus, area):
        """FIXME"""
        self.id = id
        self.node_a = node_a
        self.node_b = node_b
        self.youngs_modulus = youngs_modulus
        self.area = area

    def Dofs(self):
        """FIXME"""

        a_id = self.node_a.id
        b_id = self.node_b.id

        return [(a_id, 'u'), (a_id, 'v'), (a_id, 'w'), (b_id, 'u'), (b_id, 'v'), (b_id, 'w')]

    def Calculate(self, u, lam):
        """FIXME"""

        youngs_modulus = self.youngs_modulus
        area = self.area
        node_a = self.node_a
        node_b = self.node_b

        location_a = np.array([node_a.reference_x + u[0], node_a.reference_y + u[1], node_a.reference_z + u[2]])
        location_b = np.array([node_b.reference_x + u[3], node_b.reference_y + u[4], node_a.reference_z + u[5]])

        vector_ab = location_b - location_a

        length = la.norm(vector_ab)

        c_x = vector_ab[0] / length
        c_y = vector_ab[1] / length
        c_z = vector_ab[2] / length

        transformation_matrix = np.array([
            [ c_x**2 ,  c_x*c_y,  c_x*c_z, - c_x**2, -c_x*c_y, -c_x*c_z],
            [ c_x*c_y,   c_y**2,  c_y*c_z, -c_x*c_y, - c_y**2, -c_y*c_z],
            [ c_x*c_z,  c_y*c_z,   c_z**2, -c_x*c_z, -c_y*c_z, - c_z**2],
            [- c_x**2, -c_x*c_y, -c_x*c_z,   c_x**2,  c_x*c_y,  c_x*c_z],
            [-c_x*c_y, - c_y**2, -c_y*c_z,  c_x*c_y,   c_y**2,  c_y*c_z],
            [-c_x*c_z, -c_y*c_z, - c_z**2,  c_x*c_z,  c_y*c_z,   c_z**2],
        ])

        element_k_e = transformation_matrix * youngs_modulus * area / length
        element_k_u = None
        element_k_g = None
        element_f = None

        return element_k_e, element_k_u, element_k_g, element_f

class SingleLoad(ElementBase):
    """FIXME"""

    def __init__(self, id, node, fu, fv, fw):
        """FIXME"""
        self.id = id
        self.node = node
        self.fu = fu
        self.fv = fv
        self.fw = fw

    def Dofs(self):
        """FIXME"""

        node_id = self.node.id

        return [(node_id, 'u'), (node_id, 'v'), (node_id, 'w')]

    def Calculate(self, u, lam):
        """FIXME"""

        element_k_e = None
        element_k_u = None
        element_k_g = None
        element_f = lam * np.array([self.fu, self.fv, self.fw])

        return element_k_e, element_k_u, element_k_g, element_f

class Model(object):
    """FIXME"""

    def __init__(self, name):
        """FIXME"""

        self.name = name
        self.nodes = dict()
        self.elements = dict()
        self.dirichlet_conditions = dict()
        self.neumann_conditions = dict()
        self.previous_model = None

    def AddNode(self, id, x, y, z):
        """FIXME"""

        if id in self.nodes:
            raise RuntimeError('The model already contains a node with id {}'.format(id))

        self.nodes[id] = Node(id, x, y, z)

    def AddTrussElement(self, id, node_a, node_b, youngs_modulus, area):
        """FIXME"""

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_a not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_a))

        if node_b not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_b))

        self.elements[id] = Truss(id, self.nodes[node_a], self.nodes[node_b], youngs_modulus, area)

    def AddDirichletCondition(self, node_id, dof_types, value):
        """FIXME"""

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        for dof_type in dof_types:
            dof = (node_id, dof_type)

            if dof in self.dirichlet_conditions:
                raise RuntimeError('The model already contains a dirichlet condition for {}'.format(dof))

            self.dirichlet_conditions[dof] = value

    def AddSingleLoad(self, id, node_id, fu=0, fv=0, fw=0):
        """FIXME"""

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        self.elements[id] = SingleLoad(id, self.nodes[node_id], fu, fv, fw)

    def GetInitialModel(self):
        """FIXME"""

        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

        return current_model

    def GetModelHistory(self):
        """FIXME"""

        history = [self]

        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

            history = [current_model] + history

        return history

    def Duplicate(self):
        """FIXME"""

        self_previous_model = self.previous_model
        self.previous_model = None

        clone = deepcopy(self)

        self.previous_model = self_previous_model
        clone.previous_model = self

        return clone

    def PerformLinearSolutionStep(self, lam=1.0):
        """Just for testing"""

        model = self.Duplicate()

        model.name = f'Linear solution step (lambda={lam:.3})'

        assembler = Assembler(model)

        dof_count = assembler.dof_count

        u = np.zeros(dof_count)

        for dof, value in model.dirichlet_conditions.items():
            index = assembler.IndexOfDof(dof)
            u[index] = value

        k = np.zeros((dof_count, dof_count))
        f = np.zeros(dof_count)

        assembler.Calculate(u, lam, k, k, k, f)

        free_count = assembler.free_dof_count

        a = k[:free_count, :free_count]
        b = f[:free_count] - k[:free_count, free_count:] @ u[free_count:]

        u[:free_count] = la.solve(a, b)

        for index, dof in enumerate(assembler.dofs):
            node_id, dof_type = dof

            value = u[index]

            model.nodes[node_id].Update(dof_type, value)

        return model
