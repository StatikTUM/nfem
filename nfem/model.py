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

    def add_node(self, id, x, y, z):
        """FIXME"""

        if id in self.nodes:
            raise RuntimeError('The model already contains a node with id {}'.format(id))

        self.nodes[id] = Node(id, x, y, z)

    def add_truss_element(self, id, node_a, node_b, youngs_modulus, area):
        """FIXME"""

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_a not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_a))

        if node_b not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_b))

        self.elements[id] = Truss(id, self.nodes[node_a], self.nodes[node_b], youngs_modulus, area)

    def add_dirichlet_condition(self, node_id, dof_types, value):
        """FIXME"""

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        for dof_type in dof_types:
            dof = (node_id, dof_type)

            if dof in self.dirichlet_conditions:
                raise RuntimeError('The model already contains a dirichlet condition for {}'
                                   .format(dof))

            self.dirichlet_conditions[dof] = value

    def add_single_load(self, id, node_id, fu=0, fv=0, fw=0):
        """FIXME"""

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        self.elements[id] = SingleLoad(id, self.nodes[node_id], fu, fv, fw)

    def set_dof_state(self, dof, value):
        """FIXME"""
        node_id, dof_type = dof
        self.nodes[node_id].set_dof_state(dof_type, value)

    def get_dof_state(self, dof):
        """FIXME"""
        node_id, dof_type = dof
        return self.nodes[node_id].get_dof_state(dof_type)

    def get_initial_model(self):
        """FIXME"""

        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

        return current_model

    def get_model_history(self):
        """FIXME"""

        history = [self]

        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

            history = [current_model] + history

        return history

    def get_duplicate(self, name=None):
        """FIXME"""

        temp_previous_model = self.previous_model
        self.previous_model = None

        duplicate = deepcopy(self)

        self.previous_model = temp_previous_model
        duplicate.previous_model = self

        if name is not None:
            duplicate.name = name

        return duplicate

    def perform_linear_solution_step(self):
        """Just for testing"""

        assembler = Assembler(self)

        dof_count = assembler.dof_count

        u = np.zeros(dof_count)

        for dof, value in self.dirichlet_conditions.items():
            index = assembler.index_of_dof(dof)
            u[index] = value

        k = np.zeros((dof_count, dof_count))
        f = np.zeros(dof_count)

        assembler.assemble_matrix(k, lambda element: element.calculate_elastic_stiffness_matrix())
        assembler.assemble_vector(f, lambda element: element.calculate_external_forces())

        f *= self.lam

        free_count = assembler.free_dof_count

        a = k[:free_count, :free_count]
        b = f[:free_count] - k[:free_count, free_count:] @ u[free_count:]

        u[:free_count] = la.solve(a, b)

        for index, dof in enumerate(assembler.dofs):

            value = u[index]

            self.set_dof_state(dof, value)

    def perform_non_linear_solution(self, method=ArcLengthControl, tolerance=1e-5, max_iterations=100):
        """FIXME"""

        print("=================================")
        print("Start non linear solution step...")

        # rotate the predictor if necessary (e.g. for branch switching)
        # TODO for branch switching

        # initialize working matrices and functions for newton raphson
        assembler = Assembler(self)
        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        def calculate_system(x):
            """FIXME"""
            # update actual coordinates
            for index, dof in enumerate(assembler.free_dofs):
                value = x[index]
                self.set_dof_state(dof, value)

            # update lambda
            self.lam = x[-1]

            # initialize with zeros
            k = np.zeros((dof_count, dof_count))
            external_f = np.zeros(dof_count)
            internal_f = np.zeros(dof_count)

            # assemble stiffness
            assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

            # assemble force
            assembler.assemble_vector(external_f, lambda element: element.calculate_external_forces())
            assembler.assemble_vector(internal_f, lambda element: element.calculate_internal_forces())

            # assemble left and right hand side for newton raphson
            lhs = np.zeros((free_count + 1, free_count + 1))
            rhs = np.zeros(free_count + 1)

            # mechanical system
            lhs[:free_count, :free_count] = k[:free_count, :free_count]
            lhs[:free_count, -1] = -external_f[:free_count]
            rhs[:free_count] = internal_f[:free_count] - self.lam * external_f[:free_count]

            # constraint
            method.calculate_derivatives(self, lhs[-1, :])
            rhs[-1] = method.calculate_constraint(self)

            return lhs, rhs

        # prediction as vector for newton raphson
        x = np.zeros(free_count + 1)
        for index, dof in enumerate(assembler.free_dofs):
            x[index] = self.get_dof_state(dof)

        x[-1] = self.lam

        # solve newton raphson
        x, n_iter = NewtonRaphson(max_iterations, tolerance).solve(calculate_system, x_initial=x)

        print("Solution found after {} iteration steps.".format(n_iter))

        # TODO solve attendant eigenvalue problem

    def perform_non_linear_solution_step(self,
                                     predictor_method=LoadIncrementPredictor,
                                     path_following_method=ArcLengthControl):
        """FIXME"""

        print("=================================")
        print("Start non linear solution step...")
        # calculate the direction of the predictor
        predictor_method.predict(self)

        # rotate the predictor if necessary (e.g. for branch switching)
        # TODO for branch switching

        # scale the predictor so it fulfills the path following constraint
        path_following_method.scale_predictor(self)

        # initialize working matrices and functions for newton raphson
        assembler = Assembler(self)
        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        def calculate_system(x):
            """FIXME"""
            # update actual coordinates
            for index, dof in enumerate(assembler.free_dofs):
                value = x[index]
                self.set_dof_state(dof, value)

            # update lambda
            self.lam = x[-1]

            # initialize with zeros
            k = np.zeros((dof_count, dof_count))
            external_f = np.zeros(dof_count)
            internal_f = np.zeros(dof_count)

            # assemble stiffness
            assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

            # assemble force
            assembler.assemble_vector(external_f, lambda element: element.calculate_external_forces())
            assembler.assemble_vector(internal_f, lambda element: element.calculate_internal_forces())

            # assemble left and right hand side for newton raphson
            lhs = np.zeros((free_count + 1, free_count + 1))
            rhs = np.zeros(free_count + 1)

            # mechanical system
            lhs[:free_count, :free_count] = k[:free_count, :free_count]
            lhs[:free_count, -1] = -external_f[:free_count]
            rhs[:free_count] = internal_f[:free_count] - self.lam * external_f[:free_count]

            # constraint
            path_following_method.calculate_derivatives(self, lhs[-1, :])
            rhs[-1] = path_following_method.calculate_constraint(self)

            return lhs, rhs

        # prediction as vector for newton raphson
        x = np.zeros(free_count+1)
        for i in range(free_count):
            x[i] = self.get_dof_state(assembler.dofs[i])

        x[-1] = self.lam

        # solve newton raphson
        x, n_iter = NewtonRaphson().solve(calculate_system, x_initial=x)

        print("Solution found after {} iteration steps.".format(n_iter))

        # TODO solve attendant eigenvalue problem
