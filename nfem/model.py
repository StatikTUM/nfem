"""FIXME"""

from copy import deepcopy

import numpy as np
import numpy.linalg as la

from .node import Node
from .single_load import SingleLoad
from .truss import Truss

from .assembler import Assembler
from .newton_raphson import NewtonRaphson

from .path_following_method import ArcLengthControl
from .predictor import LoadIncrementPredictor

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
                raise RuntimeError('The model already contains a dirichlet condition for {}'
                                   .format(dof))

            self.dirichlet_conditions[dof] = value

    def AddSingleLoad(self, id, node_id, fu=0, fv=0, fw=0):
        """FIXME"""

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        self.elements[id] = SingleLoad(id, self.nodes[node_id], fu, fv, fw)

    def SetDofState(self, dof, value):
        """FIXME"""
        node_id, dof_type = dof
        self.nodes[node_id].SetDofState(dof_type, value)

    def GetDofState(self, dof):
        """FIXME"""
        node_id, dof_type = dof
        return self.nodes[node_id].GetDofState(dof_type)

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

    def GetDuplicate(self, name=None):
        """FIXME"""

        temp_previous_model = self.previous_model
        self.previous_model = None

        duplicate = deepcopy(self)

        self.previous_model = temp_previous_model
        duplicate.previous_model = self

        if name is not None:
            duplicate.name = name

        return duplicate

    def PerformLinearSolutionStep(self, lam=1.0):
        """Just for testing"""

        assembler = Assembler(self)

        self.lam = lam
        dof_count = assembler.dof_count

        u = np.zeros(dof_count)

        for dof, value in self.dirichlet_conditions.items():
            index = assembler.IndexOfDof(dof)
            u[index] = value

        k = np.zeros((dof_count, dof_count))
        f = np.zeros(dof_count)

        assembler.AssembleMatrix(k, lambda element: element.CalculateElasticStiffnessMatrix())
        assembler.AssembleVector(f, lambda element: element.CalculateExternalForces())

        f *= self.lam

        free_count = assembler.free_dof_count

        a = k[:free_count, :free_count]
        b = f[:free_count] - k[:free_count, free_count:] @ u[free_count:]

        u[:free_count] = la.solve(a, b)

        for index, dof in enumerate(assembler.dofs):

            value = u[index]

            self.SetDofState(dof, value)

    def PerformNonLinearSolutionStep(self,
                                     predictor_method=LoadIncrementPredictor,
                                     path_following_method=ArcLengthControl):
        """FIXME"""

        print("=================================")
        print("Start non linear solution step...")
        # calculate the direction of the predictor
        predictor_method.Predict(self)

        # rotate the predictor if necessary (e.g. for branch switching)
        # TODO for branch switching

        # scale the predictor so it fulfills the path following constraint
        path_following_method.ScalePredictor(self)

        # initialize working matrices and functions for newton raphson
        assembler = Assembler(self)
        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        def CalculateSystem(x):
            """FIXME"""
            # update actual coordinates
            for index, dof in enumerate(assembler.dofs[:free_count]):
                value = x[index]
                self.SetDofState(dof, value)

            # update lambda
            self.lam = x[-1]

            # initialize with zeros
            k = np.zeros((dof_count, dof_count))
            external_f = np.zeros(dof_count)
            internal_f = np.zeros(dof_count)

            # assemble stiffness
            assembler.AssembleMatrix(k, lambda element: element.CalculateStiffnessMatrix())

            # assemble force
            assembler.AssembleVector(external_f, lambda element: element.CalculateExternalForces())
            assembler.AssembleVector(internal_f, lambda element: element.CalculateInternalForces())

            # assemble left and right hand side for newton raphson
            lhs = np.zeros((free_count + 1, free_count + 1))
            rhs = np.zeros(free_count + 1)

            # mechanical system
            lhs[:free_count, :free_count] = k[:free_count, :free_count]
            lhs[:free_count, -1] = -external_f[:free_count]
            rhs[:free_count] = internal_f[:free_count] - self.lam * external_f[:free_count]

            # constraint
            path_following_method.CalculateDerivatives(self, lhs[-1, :])
            rhs[-1] = path_following_method.CalculateConstraint(self)

            return lhs, rhs

        # prediction as vector for newton raphson
        x = np.zeros(free_count+1)
        for i in range(free_count):
            x[i] = self.GetDofState(assembler.dofs[i])

        x[-1] = self.lam

        # solve newton raphson
        x, n_iter = NewtonRaphson().Solve(CalculateSystem, x_initial=x)

        print("Solution found after {} iteration steps.".format(n_iter))

        # TODO solve attendant eigenvalue problem
