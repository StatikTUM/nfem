"""This module contains the Model class.

Authors: Thomas Oberbichler, Armin Geiser
"""

from copy import deepcopy

import numpy as np
import numpy.linalg as la

from .node import Node
from .single_load import SingleLoad
from .truss import Truss

from .assembler import Assembler
from .newton_raphson import newton_raphson_solve

from .path_following_method import ArcLengthControl
from .predictor import LoadIncrementPredictor

class Model(object):
    """A Model contains all the objects that build the finite element model.
        Nodes, elements, loads, dirichlet conditions... 

    Attributes
    ----------
    name : str
        Name of the model.
    nodes : dict
        Dictionary that stores node_id : node object
    elements : str
        Dictionary that stores element_id : element object
    dirichlet_conditions : str
        Dictionary that stores dc_id : dirichlet condition object
    neumann_conditions : str
        Dictionary that stores nc_id : load object
    lam : float
        load factor
    previous_model : Model
        Previous state of this model
    """

    def __init__(self, name):
        """Create a new model.

        Parameters
        ----------
        name : str
            Name of the model.
        """

        self.name = name
        self.nodes = dict()
        self.elements = dict()
        self.dirichlet_conditions = dict()
        self.neumann_conditions = dict()
        self.lam = 0.0
        self.previous_model = None
        self.det_k = None

    def add_node(self, id, x, y, z):
        """Add a three dimensional node to the model.

        Parameters
        ----------
        id : int or str
            Unique ID of the node.
        x : float
            X coordinate.
        y : float
            Y coordinate.
        z : float
            Z coordinate.

        Examples
        --------
        Add a node with ID `B`:

        >>> model.add_node(id='B', x=5, y=2, z=0)
        """
        if id in self.nodes:
            raise RuntimeError('The model already contains a node with id {}'.format(id))

        self.nodes[id] = Node(id, x, y, z)

    def add_truss_element(self, id, node_a, node_b, youngs_modulus, area):
        """Add a three dimensional truss element to the model.

        Parameters
        ----------
        id : int or str
            Unique ID of the element.
        node_a : int or str
            ID of the first node.
        node_b : int or str
            ID of the second node.
        youngs_modulus : float
            Youngs modulus of the material for the truss.
        area : float
            Area of the cross section for the truss.

        Examples
        --------
        Add a truss element from node `A` to node `B`:

        >>> model.add_truss_element(node_a='A', node_a='B', youngs_modulus=20, area=1)
        """
        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_a not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_a))

        if node_b not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_b))

        self.elements[id] = Truss(id, self.nodes[node_a], self.nodes[node_b], youngs_modulus, area)

    def add_dirichlet_condition(self, node_id, dof_types, value):
        """Apply a dirichlet condition to the given dof types of a node.

        Parameters
        ----------
        id : int or str
            Unique ID of the element.
        node_id : int or str
            ID of the node.
        dof_types : list or str
            List with the dof types
        value : float
            Value of the boundary condition.

        Examples
        --------
        Add a support for the vertical displacement `v` at node `A`:

        >>> model.add_dirichlet_condition(node_id='A', dof_types='v', value=0)

        Lock all displacements (`u`, `v` and `w`) for a fixed support:

        >>> model.add_dirichlet_condition(node_id='B', dof_types='uvw', value=0)
        """
        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        for dof_type in dof_types:
            dof = (node_id, dof_type)

            if dof in self.dirichlet_conditions:
                raise RuntimeError('The model already contains a dirichlet condition for {}'
                                   .format(dof))

            self.dirichlet_conditions[dof] = value

    def add_single_load(self, id, node_id, fu=0, fv=0, fw=0):
        """Add a single force element to the model.

        Parameters
        ----------
        id : int or str
            Unique ID of the force element.
        node_id : int or str
            ID of the node.
        fu : float
            Load magnitude in x direction - default 0.0   
        fv : float
            Load magnitude in y direction - default 0.0 
        fw : float
            Load magnitude in z direction - default 0.0   

        Examples
        --------
        Add a single force element at node `A` with only a component in negative y direction:

        >>> model.add_single_load(id=1, node_id='A', fv=-1.0)
        """

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        self.elements[id] = SingleLoad(id, self.nodes[node_id], fu, fv, fw)

    def set_dof_state(self, dof, value):
        """Sets the state of the dof

        Parameters
        ----------
        dof : tuple(node_id, dof_type)
            Dof that is modified
        value : float
            Value that is set at the dof
        """
        node_id, dof_type = dof
        self.nodes[node_id].set_dof_state(dof_type, value)

    def get_dof_state(self, dof):
        """Sets the state of the dof

        Parameters
        ----------
        dof : tuple(node_id, dof_type)
            Dof that is asked

        Returns
        ----------
        value : float
            Value at the dof
        """
        node_id, dof_type = dof
        return self.nodes[node_id].get_dof_state(dof_type)

    def get_initial_model(self):
        """Gets the initial model of this model.

        Returns
        ----------
        model : Model
            initial model 
        """
        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

        return current_model

    def get_model_history(self):
        """Gets a list of all previous models of this model.

        Returns
        ----------
        history : list
            List of previous models
        """

        history = [self]

        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

            history = [current_model] + history

        return history

    def get_duplicate(self, name=None, branch=False):
        """Get a duplicate of the model.

        Parameters
        ----------
        name : str, optional
            Name of the new model.
        branch : bool, optional
            If `branch` is `False` the duplicate will be a successor of the current model::

                previous ----> current ----> duplicate

            If `branch` is `True` the duplicate will be a successor of the previous model::

                previous ----> current
                          \ 
                           \-> duplicate

        Returns
        -------
        model : Model
            Duplicate of the current model.
        """

        temp_previous_model = self.previous_model
        self.previous_model = None

        duplicate = deepcopy(self)

        self.previous_model = temp_previous_model

        if branch:
            duplicate.previous_model = self.previous_model
        else:
            duplicate.previous_model = self

        if name is not None:
            duplicate.name = name

        return duplicate

    def perform_linear_solution_step(self):
        """Performs a linear solution step on the model.
            It uses the member variable `lam` as load factor.
            The results are stored at the dofs and used to update the current 
            coordinates of the nodes.
        """

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

    def perform_non_linear_solution_step(self,
                                     predictor_method=LoadIncrementPredictor,
                                     path_following_method=ArcLengthControl):
        """Performs a non linear solution step on the model.
            It uses the parameter `predictor_method` to predict the solution and
            the parameter `path_following_method` to constrain the non linear problem.
            A newton raphson algorithm is used to iteratively solve the nonlinear 
            equation system r(u,lam) = 0
            The results are stored at the dofs and used to update the current 
            coordinates of the nodes.

        Parameters
        ----------
        predictor_method : Object
            Predictor object that predicts the solution. 
            (predictor.py for details)            
        path_following_method : Object
            Path following object that constrains the solution. 
            (path_following_method.py for details)
        """

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
            """Callback function for the newton raphson method that calculates the 
                system for a given state x

            Parameters
            ----------
            x : numpy.ndarray
                Current state of dofs and lambda (unknowns of the non linear system) 

            Returns
            ----------
            lhs : numpy.ndarray
                Left hand side matrix of size (n_free_dofs+1,n_free_dofs+1).
                Containts the derivatives of the residuum and the constraint.
            rhs : numpy.ndarray
                Right hand side vector of size (n_free_dofs+1).
                Contains the values of the residuum of the structure and the constraint.
            """

            # update actual coordinates
            for index, dof in enumerate(assembler.dofs[:free_count]):
                value = x[index]
                self.set_dof_state(dof, value)

            # update lambda
            self.lam = x[-1]

            # initialize matrices and vectors with zeros
            k = np.zeros((dof_count, dof_count))
            external_f = np.zeros(dof_count)
            internal_f = np.zeros(dof_count)

            # assemble stiffness
            assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

            # assemble force
            assembler.assemble_vector(external_f, lambda element: element.calculate_external_forces())
            assembler.assemble_vector(internal_f, lambda element: element.calculate_internal_forces())

            # initialize left and right hand side for newton raphson
            lhs = np.zeros((free_count + 1, free_count + 1))
            rhs = np.zeros(free_count + 1)

            # assemble contribution from mechanical system
            lhs[:free_count, :free_count] = k[:free_count, :free_count]
            lhs[:free_count, -1] = -external_f[:free_count]
            rhs[:free_count] = internal_f[:free_count] - self.lam * external_f[:free_count]

            # assemble contribution from constraint
            path_following_method.calculate_derivatives(self, lhs[-1, :])
            rhs[-1] = path_following_method.calculate_constraint(self)

            # solve det(k)
            self.det_k = la.det(k[:free_count, :free_count])

            return lhs, rhs

        # initialize prediction vector for newton raphson
        x = np.zeros(free_count+1)

        # assemble contribution from dofs
        for i in range(free_count):
            x[i] = self.get_dof_state(assembler.dofs[i])

        # assemble contribution from lambda
        x[-1] = self.lam

        # solve newton raphson
        x, n_iter = newton_raphson_solve(calculate_system, x_initial=x)

        print("Solution found after {} iteration steps.".format(n_iter))

        # TODO solve attendant eigenvalue problem

        return
