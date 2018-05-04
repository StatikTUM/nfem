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

from .path_following_method import ArcLengthControl, DisplacementControl, LoadControl

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
        self._nodes = dict()
        self._elements = dict()
        self.dirichlet_conditions = dict()
        self.neumann_conditions = dict()
        self.lam = 0.0
        self.previous_model = None

    @property
    def nodes(self):
        """Get a list of all nodes in the model.

        Returns
        -------
        nodes : list
            List of all nodes in the model.
        """
        return self._nodes.values()

    def get_node(self, id):
        """Get a node by its ID.

        Parameters
        ----------
        id : int or str
            ID of the node.

        Returns
        -------
        node : list
            Node with the given ID.
        """
        return self._nodes[id]

    @property
    def elements(self):
        """Get a list of all elements in the model.

        Returns
        -------
        elements : list
            List of all elements in the model.
        """
        return self._elements.values()

    def get_element(self, id):
        """Get an element by its ID.

        Parameters
        ----------
        id : int or str
            ID of the element.

        Returns
        -------
        element : list
            Element with the given ID.
        """
        return self._elements[id]

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
        if id in self._nodes:
            raise RuntimeError('The model already contains a node with id {}'.format(id))

        self._nodes[id] = Node(id, x, y, z)

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
        if id in self._elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_a not in self._nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_a))

        if node_b not in self._nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_b))

        self._elements[id] = Truss(id, self._nodes[node_a], self._nodes[node_b], youngs_modulus, area)

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
        if node_id not in self._nodes:
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

        if id in self._elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_id not in self._nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        self._elements[id] = SingleLoad(id, self._nodes[node_id], fu, fv, fw)

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
        self._nodes[node_id].set_dof_state(dof_type, value)

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
        return self._nodes[node_id].get_dof_state(dof_type)

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
    
    # === solve functions

    #======================================
    # Solution functions
    #======================================

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

    def perform_non_linear_solution_step(self, strategy, tolerance=1e-5, max_iterations=100, **options):
        """Performs a non linear solution step on the model.
            The path following strategy is chose according to the parameter.
            A newton raphson algorithm is used to iteratively solve the nonlinear 
            equation system r(u,lam) = 0
            The results are stored at the dofs and used to update the current 
            coordinates of the nodes.

        Parameters
        ----------
        strategy : string
            Path following strategy. Available options:
            - load-control 
            - displacement-control
            - arc-length-control
        max_iterations: int
            Maximum number of iteration for the newton raphson
        tolerance : float
            Tolerance for the newton raphson
        **options: kwargs (key word arguments)
            Additional options e.g. 
            "dof=('B','v')" for displacement-control
        """

        print("=================================")
        print("Start non linear solution step...")

        if strategy == 'load-control':
            constraint = LoadControl(self)
        elif strategy == 'displacement-control':
            constraint = DisplacementControl(self, **options)
        elif strategy == 'arc-length-control':
            constraint = ArcLengthControl(self)
        else:
            raise ValueError('Invalid path following strategy:' + strategy)

        # initialize working matrices and functions for newton raphson
        assembler = Assembler(self)
        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        def calculate_system(x):
            """Callback function for the newton raphson method that calculates the 
                system for a given state x

            Parameters
            ----------
            x : ndarray
                Current state of dofs and lambda (unknowns of the non linear system) 

            Returns
            ----------
            lhs : ndarray
                Left hand side matrix of size (n_free_dofs+1,n_free_dofs+1).
                Containts the derivatives of the residuum and the constraint.
            rhs : ndarray
                Right hand side vector of size (n_free_dofs+1).
                Contains the values of the residuum of the structure and the constraint.
            """

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

            # assemble contribution from constraint
            constraint.calculate_derivatives(self, lhs[-1, :])
            rhs[-1] = constraint.calculate_constraint(self)

            return lhs, rhs

        # prediction as vector for newton raphson
        x = np.zeros(free_count + 1)
        for index, dof in enumerate(assembler.free_dofs):
            x[index] = self.get_dof_state(dof)

        x[-1] = self.lam

        # solve newton raphson
        x, n_iter = newton_raphson_solve(calculate_system, x, max_iterations, tolerance)

        print("Solution found after {} iteration steps.".format(n_iter))

    # === prediction functions

    def predict_load_factor(self, value):
        """Predicts the solution by predictor_method lambda

        Parameters
        ----------
        value : float
            Value for the new load factor lambda.  
        """
        self.lam = value

    def predict_load_increment(self, value):
        """Predicts the solution by incrementing lambda

        Parameters
        ----------
        value : float
            Value that is used to increment the load factor lambda. 
        """
        self.lam += value

    def predict_dof_state(self, dof, value):
        """Predicts the solution by predictor_method the dof

        Parameters
        ----------
        dof : object
            Dof that is prescribed.
        value : float
            Value that is used to prescribe the dof.
        """
        self.set_dof_state(dof, value)

    def predict_dof_increment(self, dof, value):
        """Predicts the solution by incrementing the dof

        Parameters
        ----------
        dof : object
            Dof that is incremented.
        value : float
            Value that is used to increment the dof.
        """
        tmp_value = self.get_dof_state(dof) + value
        self.set_dof_state(dof, tmp_value)
    
    def predict_with_last_increment(self):
        """Predicts the solution by incrementing lambda and all dofs with the 
           increment of the last solution step

        Raises
        ------
        RuntimeError
            If the model has not already one calculated step.
        """
        if self.previous_model.previous_model == None:
            raise RuntimeError('predict_with_last_increment can only be used after the first step!')

        assembler = Assembler(self)

        last_increment = self.previous_model.get_increment_vector(assembler)

        # update dofs at model
        for index, dof in enumerate(assembler.dofs):

            old_value = self.get_dof_state(dof)
            value = last_increment[index]

            self.set_dof_state(dof, old_value + value)

        # update lam at model
        self.lam += last_increment[-1]


    def predict_tangential(self, predictor_method, **options):
        """Predicts the solution by incrementing lambda and all dofs with the 
           increment of the last solution step
        """
        # solve tangend vector
        assembler = Assembler(self)
        t = self.solve_tangent_vector(assembler=assembler)

        # caclulate scaling factor acoording to chosen prescription
        factor = 1.0
        if predictor_method == 'lambda':
            prescribed_lam = options['value']
            delta_lambda = prescribed_lam - self.lam
            factor = delta_lambda
        elif predictor_method == 'delta-lambda':
            delta_lambda = options['value']
            factor = delta_lambda
        elif predictor_method == 'u':
            dof = options['dof']
            u_old = self.get_dof_state(dof)
            u_prescribed = options['value']
            delta_u_prescribed = u_prescribed- u_old
            dof_index = assembler.index_of_dof(dof)
            delta_u_current = t[dof_index]
            factor = delta_u_prescribed/delta_u_current
        elif predictor_method == 'delta-u':
            dof = options['dof']
            delta_u_prescribed = options['value']
            dof_index = assembler.index_of_dof(dof)
            delta_u_current = t[dof_index]
            factor = delta_u_prescribed/delta_u_current
        elif predictor_method == 'arc-length':
            if self.previous_model.previous_model == None:
                raise RuntimeError('Tangential arc-length predictor can only be used after the first step!')
            previous_increment = self.previous_model.get_increment_vector(assembler)
            prescribed_l = la.norm(previous_increment)
            current_l = la.norm(t)
            factor = prescribed_l/current_l
            # make sure the tangent points in a similar direction as the last increment
            sign = np.sign(previous_increment @ t)
            if sign != 0.0:
                factor *= sign
        else:            
            raise ValueError('Invalid predictor_method for prediction:' + predictor_method)

        # scale tangent vector
        t *= factor
        
        # update dofs at model
        for index, dof in enumerate(assembler.dofs):

            old_value = self.get_dof_state(dof)
            value = t[index]

            self.set_dof_state(dof, old_value + value)

        # update lam at model
        self.lam += t[-1]

        return


    def get_increment_vector(self, assembler=None):
        """gets the increment that resulted in the current position
        """      
        if assembler == None:
            assembler = Assembler(self)  
        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count
 
        increment = np.zeros(dof_count+1)
 
        if self.previous_model == None:
            print("WARNING: increment is zero because no previous model exists!")
            return increment

        for index, dof in enumerate(assembler.dofs):
            increment[index] = self.get_dof_state(dof) - self.previous_model.get_dof_state(dof)

        increment[-1] = self.lam - self.previous_model.lam

        return increment


    def solve_tangent_vector(self, assembler=None):
        """solves for the tangent vector t=[v, 1]
            with v = d_u/d_lambda ... incremental velocity
        """      
        if assembler == None:
            assembler = Assembler(self)  
        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count
 
        v = np.zeros(dof_count)
 
        for dof, value in self.dirichlet_conditions.items():
            index = assembler.index_of_dof(dof)
            v[index] = value

        k = np.zeros((dof_count, dof_count))
        external_f = np.zeros(dof_count)

        # assemble stiffness
        assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

        # assemble force
        assembler.assemble_vector(external_f, lambda element: element.calculate_external_forces())    

        a = k[:free_count, :free_count]
        b = external_f[:free_count] - k[:free_count, free_count:] @ v[free_count:]

        v[:free_count] = la.solve(a, b)

        return np.append(v, 1.0)