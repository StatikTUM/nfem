"""FIXME"""

from copy import deepcopy

import numpy as np
import numpy.linalg as la

from .assembler import Assembler
from .newton_raphson import NewtonRaphson

from .path_following_method import LoadControl
from .predictor import LoadIncrementPredictor

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

    def GetReferenceLocation(self):
        return np.array([
            self.reference_x,
            self.reference_y,
            self.reference_z
        ])

    def Update(self, dof_type, value):
        """FIXME"""

        if dof_type == 'u':
            self.x += value
        elif dof_type == 'v':
            self.y += value
        elif dof_type == 'w':
            self.z += value
        else:
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

    def CalculateElasticStiffnessMatrix(self):
        """FIXME"""

        location_a = self.node_a.GetReferenceLocation()
        location_b = self.node_b.GetReferenceLocation()
        EA = self.youngs_modulus * self.area

        dx, dy, dz = location_b - location_a

        L = la.norm([dx, dy, dz])
        L3 = L**3

        k_e = np.empty((6, 6))

        k_e[0, 0] = (EA * dx * dx) / L3
        k_e[0, 1] = (EA * dx * dy) / L3
        k_e[0, 2] = (EA * dx * dz) / L3
        k_e[0, 3] = -k_e[0, 0]
        k_e[0, 4] = -k_e[0, 1]
        k_e[0, 5] = -k_e[0, 2]
        k_e[1, 1] = (EA * dy * dy) / L3
        k_e[1, 2] = (EA * dy * dz) / L3
        k_e[1, 3] = k_e[0, 4]
        k_e[1, 4] = -k_e[1, 1]
        k_e[1, 5] = -k_e[1, 2]
        k_e[2, 2] = (EA * dz * dz) / L3
        k_e[2, 3] = -k_e[0, 2]
        k_e[2, 4] = -k_e[1, 2]
        k_e[2, 5] = -k_e[2, 2]
        k_e[3, 3] = k_e[0, 0]
        k_e[3, 4] = k_e[0, 1]
        k_e[3, 5] = k_e[0, 2]
        k_e[4, 4] = k_e[1, 1]
        k_e[4, 5] = k_e[1, 2]
        k_e[5, 5] = k_e[2, 2]

        # symmetry

        k_e[1, 0] = k_e[0, 1]
        k_e[2, 0] = k_e[0, 2]
        k_e[2, 1] = k_e[1, 2]
        k_e[3, 0] = k_e[0, 3]
        k_e[3, 1] = k_e[1, 3]
        k_e[3, 2] = k_e[2, 3]
        k_e[4, 0] = k_e[0, 4]
        k_e[4, 1] = k_e[1, 4]
        k_e[4, 2] = k_e[2, 4]
        k_e[4, 3] = k_e[3, 4]
        k_e[5, 0] = k_e[0, 5]
        k_e[5, 1] = k_e[1, 5]
        k_e[5, 2] = k_e[2, 5]
        k_e[5, 3] = k_e[3, 5]
        k_e[5, 4] = k_e[4, 5]

        return k_e

    def CalculateGeometricStiffnessMatrix(self, u):
        """FIXME"""

        location_a = self.node_a.GetReferenceLocation()
        location_b = self.node_b.GetReferenceLocation()
        E = self.youngs_modulus
        A = self.area

        prestress = 0

        du, dv, dw = u[3:] - u[3:]
        dx, dy, dz = location_b - location_a

        L = la.norm([dx, dy, dz])
        l = la.norm([dx + du, dy + dv, dz + dw])

        e_gl = (l**2 - L**2) / (2.00 * L**2)
        L3 = L**3

        K_sigma = ((E * A * e_gl) / L) + ((prestress * A) / L)
        K_uij = (E * A) / L3

        k_g = np.empty((6, 6))

        k_g[0, 0] = K_sigma + K_uij * (2 * du * dx + du * du)
        k_g[0, 1] = K_uij * (dx * dv + dy * du + du * dv)
        k_g[0, 2] = K_uij * (dx * dw + dz * du + du * dw)
        k_g[0, 3] = -k_g[0, 0]
        k_g[0, 4] = -k_g[0, 1]
        k_g[0, 5] = -k_g[0, 2]
        k_g[1, 1] = K_sigma + K_uij * (2 * dv * dy + dv * dv)
        k_g[1, 2] = K_uij * (dy * dw + dz * dv + dv * dw)
        k_g[1, 3] = k_g[0, 4]
        k_g[1, 4] = -k_g[1, 1]
        k_g[1, 5] = -k_g[1, 2]
        k_g[2, 2] = K_sigma + K_uij * (2 * dw * dz + dw * dw)
        k_g[2, 3] = -k_g[0, 2]
        k_g[2, 4] = -k_g[1, 2]
        k_g[2, 5] = -k_g[2, 2]
        k_g[3, 3] = k_g[0, 0]
        k_g[3, 4] = k_g[0, 1]
        k_g[3, 5] = k_g[0, 2]
        k_g[4, 4] = k_g[1, 1]
        k_g[4, 5] = k_g[1, 2]
        k_g[5, 5] = k_g[2, 2]

        # symmetry

        k_g[1, 0] = k_g[0, 1]
        k_g[2, 0] = k_g[0, 2]
        k_g[2, 1] = k_g[1, 2]
        k_g[3, 0] = k_g[0, 3]
        k_g[3, 1] = k_g[1, 3]
        k_g[3, 2] = k_g[2, 3]
        k_g[4, 0] = k_g[0, 4]
        k_g[4, 1] = k_g[1, 4]
        k_g[4, 2] = k_g[2, 4]
        k_g[4, 3] = k_g[3, 4]
        k_g[5, 0] = k_g[0, 5]
        k_g[5, 1] = k_g[1, 5]
        k_g[5, 2] = k_g[2, 5]
        k_g[5, 3] = k_g[3, 5]
        k_g[5, 4] = k_g[4, 5]

        return k_g

    def Calculate(self, u, lam):
        """FIXME"""

        element_k_e = self.CalculateElasticStiffnessMatrix()
        element_k_u = None
        element_k_g = self.CalculateGeometricStiffnessMatrix(u)
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
        self.lam = 0.0
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

    def PerformNonLinearSolutionStep(self, 
                                     path_following_class=LoadControl, 
                                     predictor_class=LoadIncrementPredictor, 
                                     prescribed_value=1.0 ):
        """Currently hardcoded for LoadControl"""

        print(LoadControl)
        print(path_following_class)

        path_following_method = path_following_class(prescribed_value)

        # create a model for the predictor
        predictor_model = self.Duplicate()
        predictor_model.internal_flag = True

        # calculate the direction of the predictor
        predictor_model = predictor_class().Predict(predictor_model)

        # rotate the predictor if necessary (e.g. for branch switching)

        # scale the predictor so it fulfills the path following constraint
        predictor_model = path_following_method.ScalePredictor(predictor_model)

        # initialize working matrices and functions for newton raphson
        model = predictor_model
        assembler = Assembler(model)
        dof_count = assembler.dof_count
        LHS = np.zeros((dof_count+1,dof_count+1))
        RHS = np.zeros(dof_count+1)
        x = np.zeros(dof_count+1) # TODO assemble from model: u = x-reference_x
        def UpdateModel(x):
            pass # store a updated model if necessary
        def Compute_LHS(x):
            k = LHS[:-1]
            u = x[:-1]
            lam = x[-1]
            f = np.zeros(dof_count)
            assembler.Calculate(u, lam, k, k, k , f)
            return LHS
        def Compute_RHS(x):
            k = LHS[:-1]
            u = x[:-1]
            lam = x[-1]
            f = np.zeros(dof_count)
            assembler.Calculate(u, lam, k, k, k, f)
            print (k,u,f)
            RHS = (k*u)-f
            return RHS 

        # solve newton raphson
        x = NewtonRaphson().Solve(UpdateModel, Compute_LHS, Compute_RHS, x_initial=x)

        # update model (maybe this should happen already in newton raphson)
        # TODO

        # remove iterations from history
        # TODO

        model.name = f'Non linear solution step (lambda={lam:.3})'
        return model