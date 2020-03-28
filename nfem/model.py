"""This module contains the Model class.

Authors: Thomas Oberbichler, Armin Geiser
"""

from copy import deepcopy
from enum import Enum

import numpy as np
import numpy.linalg as la

from scipy.linalg import eig

from nfem.dof import Dof
from nfem.node import Node
from nfem.truss import Truss

from nfem.assembler import Assembler
from nfem.newton_raphson import newton_raphson_solve

from nfem.path_following_method import ArcLengthControl, DisplacementControl, LoadControl


class ModelStatus(Enum):
    """Enum for the model status """
    initial = 0
    duplicate = 1
    prediction = 2
    iteration = 3
    equilibrium = 4
    eigenvector = 5


class CompletionsView:
    def __init__(self, dictionary):
        self._dictionary = dictionary

    def __getitem__(self, key):
        return self._dictionary[key]

    def __len__(self):
        return self._dictionary.values().__len__()

    def __iter__(self):
        return self._dictionary.values().__iter__()

    def __next__(self):
        return self._dictionary.values().__next__()

    def _ipython_key_completions_(self):
        return list(self._dictionary.keys())


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
    lam : float
        load factor
    previous_model : Model
        Previous state of this model
    """

    def __init__(self, name=None):
        """Create a new model.

        Parameters
        ----------
        name : str
            Name of the model.
        """

        self.name = name
        self.status = ModelStatus.initial
        self._nodes = dict()
        self._elements = dict()
        self.lam = 0.0
        self._previous_model = None
        self.det_k = None
        self.first_eigenvalue = None
        self.first_eigenvector_model = None

    def get_previous_model(self, skip_iterations=True):
        """Get the previous model of the current model.

        Parameters
        ----------
        skip_iterations : bool
            Flag if iteration or predicted previous models should be skipped

        Returns
        -------
        model : Model
            The previous model object
        """
        if not skip_iterations:
            return self._previous_model

        # find the most previous model that is not an iteration or prediction
        previous_model = self._previous_model

        while previous_model is not None and previous_model.status in [ModelStatus.duplicate,
                                                                       ModelStatus.prediction, ModelStatus.iteration]:
            previous_model = previous_model._previous_model

        return previous_model

    @property
    def load_factor(self):
        return self.lam

    @load_factor.setter
    def load_factor(self, value):
        self.lam = value

    @property
    def nodes(self):
        """Get a list of all nodes in the model.

        Returns
        -------
        nodes : list
            List of all nodes in the model.
        """
        return CompletionsView(self._nodes)

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
        return CompletionsView(self._elements)

    @property
    def structural_elements(self):
        """
        FIXME
        """
        elements = []
        for element in self.elements:
            if isinstance(element, Truss):
                elements.append(element)
        return elements

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

    @property
    def free_dofs(self):
        return Assembler(self).free_dofs

    # === modeling

    def add_node(self, id, x, y, z, support='', fx=0.0, fy=0.0, fz=0.0):
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

        Returns
        ----------
        Node
            The new Node

        Examples
        --------
        Add a node with ID `B`:

        >>> model.add_node(id='B', x=5, y=2, z=0)
        """
        if id in self._nodes:
            raise RuntimeError('The model already contains a node with id {}'.format(id))

        node = Node(id, x, y, z)

        self._nodes[id] = node

        if 'x' in support:
            node.dof('u').is_active = False
        if 'y' in support:
            node.dof('v').is_active = False
        if 'z' in support:
            node.dof('w').is_active = False

        node.dof('u').external_force = fx
        node.dof('v').external_force = fy
        node.dof('w').external_force = fz

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

        Returns
        ----------
        Truss
            The new Truss

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

        element = Truss(id, self._nodes[node_a], self._nodes[node_b], youngs_modulus, area)

        self._elements[id] = element

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

        node = self.get_node(node_id)

        for dof_type in dof_types:
            dof = node.dof(dof_type)
            dof.is_active = False
            dof.delta = value

    # === degree of freedoms

    def __getitem__(self, key):
        if isinstance(key, Dof):
            node_key, dof_type = key.key
        else:
            node_key, dof_type = key
        return self.get_node(node_key).dof(dof_type)

    # === increment

    def get_dof_increment(self, dof):
        """Get the increment of the dof during the last solution step

        Parameters
        ----------
        dof : tuple(node_id, dof_type)
            Dof that is asked

        Returns
        -------
        delta : float
            Increment of the dof during the last step
        """
        if self.get_previous_model() is None:
            return 0.0

        previous_model = self.get_previous_model()

        current_value = self[dof].delta
        previous_value = previous_model[dof].delta

        return current_value - previous_value

    def get_lam_increment(self):
        """Get the increment of lambda during the last solution step

        Returns
        -------
        delta : float
            Increment of lambda during the last step
        """
        if self.get_previous_model() is None:
            return 0.0

        current_value = self.lam
        previous_value = self.get_previous_model().lam

        return current_value - previous_value

    def get_increment_vector(self, assembler=None):
        """Get the increment that resulted in the current position
        """

        if assembler is None:
            assembler = Assembler(self)

        dof_count = assembler.dof_count

        increment = np.zeros(dof_count + 1)

        if self.get_previous_model() is None:
            print('WARNING: Increment is zero because no previous model exists!')
            return increment

        for index, dof in enumerate(assembler.dofs):
            increment[index] = self.get_dof_increment(dof)

        increment[-1] = self.get_lam_increment()

        return increment

    def get_increment_norm(self, assembler=None):
        increment = self.get_increment_vector(assembler)
        return la.norm(increment)

    # === model history

    def get_initial_model(self):
        """Gets the initial model of this model.

        Returns
        ----------
        model : Model
            initial model
        """
        current_model = self

        while current_model.get_previous_model() is not None:
            current_model = current_model.get_previous_model()

        return current_model

    def get_model_history(self, skip_iterations=True):
        """Gets a list of all previous models of this model.

        Parameters
        ----------
        skip_iterations : bool
            Flag that decides if non converged previous models should be considered

        Returns
        ----------
        history : list
            List of previous models
        """

        history = [self]

        current_model = self

        while current_model.get_previous_model(skip_iterations) is not None:
            current_model = current_model.get_previous_model(skip_iterations)

            history = [current_model] + history

        return history

    def get_duplicate(self, name=None, branch=False):
        r"""Get a duplicate of the model.

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

        temp_previous_model = self._previous_model
        self._previous_model = None

        duplicate = deepcopy(self)

        self._previous_model = temp_previous_model

        if branch:
            duplicate._previous_model = self._previous_model
        else:
            duplicate._previous_model = self
            duplicate.status = ModelStatus.duplicate

        if name is not None:
            duplicate.name = name

        # make sure the duplicated model is in a clean state
        duplicate.det_k = None
        duplicate.first_eigenvalue = None
        duplicate.first_eigenvector_model = None

        return duplicate

    def new_timestep(self, name=None):
        temp_previous_model = self._previous_model
        self._previous_model = None

        duplicate = deepcopy(self)

        self._previous_model = temp_previous_model

        duplicate._previous_model = self

        if name is not None:
            duplicate.name = name

        duplicate.det_k = None
        duplicate.first_eigenvalue = None
        duplicate.first_eigenvector_model = None

        return duplicate

    # === solving

    def perform_linear_solution_step(self):
        """Performs a linear solution step on the model.
            It uses the member variable `lam` as load factor.
            The results are stored at the dofs and used to update the current
            coordinates of the nodes.
        """

        print("\n=================================")
        print("Start linear solution step...")
        print("lambda : {}".format(self.lam))

        assembler = Assembler(self)

        dof_count = assembler.dof_count

        u = np.zeros(dof_count)

        for dof in assembler.dofs[assembler.free_dof_count:]:
            index = assembler.index_of_dof(dof)
            u[index] = self[dof].delta

        k = np.zeros((dof_count, dof_count))
        f = np.zeros(dof_count)

        for i, dof in enumerate(assembler.free_dofs):
            f[i] += self[dof].external_force

        assembler.assemble_matrix(k, lambda element: element.calculate_elastic_stiffness_matrix())
        assembler.assemble_vector(f, lambda element: element.calculate_external_forces())

        f *= self.lam

        free_count = assembler.free_dof_count

        a = k[:free_count, :free_count]
        b = f[:free_count] - k[:free_count, free_count:] @ u[free_count:]

        u[:free_count] = la.solve(a, b)

        for index, dof in enumerate(assembler.dofs):
            self[dof].delta = u[index]

        self.status = ModelStatus.equilibrium

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
            - dof=('B','v'): for displacement-control
            - solve_det_k=True: for solving the determinant of k at convergence
            - solve_attendant_eigenvalue=True: for solving the attendant eigenvalue problem at convergence
        """

        print("\n=================================")
        print("Start non linear solution step...")

        if strategy == 'load-control':
            constraint = LoadControl(self)
        elif strategy == 'displacement-control':
            constraint = DisplacementControl(self, options['dof'])
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

            # create a duplicate of the current state before updating and insert it in the history
            duplicate = self.get_duplicate()
            duplicate._previous_model = self._previous_model
            self._previous_model = duplicate
            duplicate.status = self.status

            # update status flag
            self.status = ModelStatus.iteration

            # update actual coordinates
            for index, dof in enumerate(assembler.free_dofs):
                self[dof].delta = x[index]

            # update lambda
            self.lam = x[-1]

            # initialize with zeros
            k = np.zeros((dof_count, dof_count))
            external_f = np.zeros(dof_count)
            internal_f = np.zeros(dof_count)

            # assemble stiffness
            assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

            # assemble force

            for i, dof in enumerate(assembler.free_dofs):
                external_f[i] += self[dof].external_force

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
            x[index] = self[dof].delta

        x[-1] = self.lam

        # solve newton raphson
        x, n_iter = newton_raphson_solve(calculate_system, x, max_iterations, tolerance)

        print("Solution found after {} iteration steps.".format(n_iter))

        self.status = ModelStatus.equilibrium

        if 'solve_det_k' in options and not options['solve_det_k']:
            pass
        else:
            self.solve_det_k(assembler=assembler)

        if 'solve_attendant_eigenvalue' in options:
            if options['solve_attendant_eigenvalue']:
                self.solve_eigenvalues(assembler=assembler)

    def solve_det_k(self, k=None, assembler=None):
        """Solves the determinant of k

        Parameters
        ----------
        k : numpy.ndarray (optional)
            stiffness matrix can be directly passed.
        assembler : Object (optional)
            assembler can be passed to speed up if k is not given
        """
        if k is None:
            if assembler is None:
                assembler = Assembler(self)
            dof_count = assembler.dof_count
            free_count = assembler.free_dof_count
            k = np.zeros((dof_count, dof_count))
            assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())
        self.det_k = la.det(k[:free_count, :free_count])
        print("Det(K): {}".format(self.det_k))

    def solve_linear_eigenvalues(self, assembler=None):
        """Solves the linearized eigenvalue problem
           [ k_e + eigvals * k_g(linear strain) ] * eigvecs = 0
           Stores the first positive eigenvalue and vector

        Parameters
        ----------
        assembler : Object (optional)
            assembler can be passed to speed up
        """
        if assembler is None:
            assembler = Assembler(self)

        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        # assemble matrices
        k_e = np.zeros((dof_count, dof_count))
        k_g = np.zeros((dof_count, dof_count))
        print("=================================")
        print('Linearized prebuckling (LPB) analysis ...')
        assembler.assemble_matrix(k_e, lambda element: element.calculate_elastic_stiffness_matrix())
        assembler.assemble_matrix(k_g, lambda element: element.calculate_geometric_stiffness_matrix(linear=True))

        # solve eigenvalue problem
        eigvals, eigvecs = eig((k_e[:free_count, :free_count]), -k_g[:free_count, :free_count])

        # extract real parts of eigenvalues
        eigvals = np.array([x.real for x in eigvals])

        # sort eigenvalues and vectors
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # remove negative eigenvalues
        for i, eigenvalue in enumerate(eigvals):
            if eigenvalue > 0.0:
                break
            else:
                i += 1

        if i == len(eigvals):
            print('System has no positive eigenvalues!')
            return

        eigvals = eigvals[i:]
        eigvecs = eigvecs[:, i:]

        print('First linear eigenvalue: {}'.format(eigvals[0]))
        print('First linear eigenvalue * lambda: {}'.format(eigvals[0] * self.lam))  # this is printed in TRUSS
        if len(eigvecs[0]) < 10:
            print('First linear eigenvector: {}'.format(eigvecs[0]))

        self.first_eigenvalue = eigvals[0]

        # store eigenvector as model
        model = self.get_duplicate()
        model._previous_model = self
        model.status = ModelStatus.eigenvector
        model.det_k = None
        model.first_eigenvalue = None
        model.first_eigenvector_model = None
        model.lam = None

        for index, dof in enumerate(assembler.free_dofs):
            model[dof].delta = eigvecs[index][0]

        self.first_eigenvector_model = model

    def solve_eigenvalues(self, assembler=None):
        """Solves the eigenvalue problem
           [ k_m + eigvals * k_g ] * eigvecs = 0
           Stores the closest (most critical) eigenvalue and vector

        Parameters
        ----------
        assembler : Object (optional)
            assembler can be passed to speed up
        """
        if assembler is None:
            assembler = Assembler(self)

        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        # assemble matrices
        k_m = np.zeros((dof_count, dof_count))
        k_g = np.zeros((dof_count, dof_count))
        print("=================================")
        print('Attendant eigenvalue analysis ...')
        assembler.assemble_matrix(k_m, lambda element: element.calculate_material_stiffness_matrix())
        assembler.assemble_matrix(k_g, lambda element: element.calculate_geometric_stiffness_matrix())

        # solve eigenvalue problem
        eigvals, eigvecs = eig((k_m[:free_count, :free_count]), -k_g[:free_count, :free_count])

        # extract real parts of eigenvalues
        eigvals = np.array([x.real for x in eigvals])

        # sort eigenvalues and vectors
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # find index of closest eigenvalue to 1 (we could store all but that seems like an overkill)
        idx = (np.abs(eigvals - 1.0)).argmin()

        print('Closest eigenvalue: {}'.format(eigvals[idx]))
        print('Closest eigenvalue * lambda: {}'.format(eigvals[idx] * self.lam))  # this is printed in TRUSS
        if len(eigvecs[idx]) < 10:
            print('Closest eigenvector: {}'.format(eigvecs[idx]))

        self.first_eigenvalue = eigvals[idx]

        # store eigenvector as model
        model = self.get_duplicate()
        model._previous_model = self
        model.status = ModelStatus.eigenvector
        model.det_k = None
        model.first_eigenvalue = None
        model.first_eigenvector_model = None
        model.lam = None

        for index, dof in enumerate(assembler.free_dofs):
            model[dof].delta = eigvecs[index][idx]

        self.first_eigenvector_model = model

    def get_tangent_vector(self, assembler=None):
        """ Get the tangent vector

        Returns
        -------
        tangent : ndarray
            Tangent vector t = [v, 1]
            with v = d_u / d_lambda ... incremental velocity
        """
        if assembler is None:
            assembler = Assembler(self)

        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        tangent = np.zeros(dof_count + 1)

        v = tangent[:-1]

        for dof in assembler.dofs[assembler.free_dof_count:]:
            index = assembler.index_of_dof(dof)
            v[index] = self[dof].delta

        k = np.zeros((dof_count, dof_count))
        external_f = np.zeros(dof_count)

        # assemble stiffness
        assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

        # assemble force

        for i, dof in enumerate(assembler.free_dofs):
            external_f[i] += self[dof].external_force

        assembler.assemble_vector(external_f, lambda element: element.calculate_external_forces())

        lhs = k[:free_count, :free_count]
        rhs = external_f[:free_count] - k[:free_count, free_count:] @ v[free_count:]

        v[:free_count] = la.solve(lhs, rhs)

        # lambda = 1
        tangent[-1] = 1

        return tangent

    # === prediction functions

    def predict_load_factor(self, value):
        """Predicts the solution by predictor_method lambda

        Parameters
        ----------
        value : float
            Value for the new load factor lambda.
        """
        self.status = ModelStatus.prediction
        self.lam = value

    def predict_load_increment(self, value):
        """Predicts the solution by incrementing lambda

        Parameters
        ----------
        value : float
            Value that is used to increment the load factor lambda.
        """
        self.status = ModelStatus.prediction
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
        self.status = ModelStatus.prediction
        self[dof].delta = value

    def predict_dof_increment(self, dof, value):
        """Predicts the solution by incrementing the dof

        Parameters
        ----------
        dof : object
            Dof that is incremented.
        value : float
            Value that is used to increment the dof.
        """
        self.status = ModelStatus.prediction
        self[dof].delta += value

    def predict_with_last_increment(self, value=None):
        """Predicts the solution by incrementing lambda and all dofs with the
           increment of the last solution step

        Parameters
        ----------
        value : float (optional)
            Length of the increment

        Raises
        ------
        RuntimeError
            If the model has not already one calculated step.
        """
        self.status = ModelStatus.prediction
        if self.get_previous_model().get_previous_model() is None:
            raise RuntimeError('predict_with_last_increment can only be used after the first step!')

        assembler = Assembler(self)

        last_increment = self.get_previous_model().get_increment_vector(assembler)

        length = la.norm(last_increment)

        if value is not None and length != 0.0:
            last_increment *= value/length
            length *= value

        if length == 0.0:
            print("WARNING: The length of the prescribed increment is 0.0!")

        # update dofs at model
        for index, dof in enumerate(assembler.dofs):
            self[dof].delta += last_increment[index]

        # update lam at model
        self.lam += last_increment[-1]

    def predict_tangential(self, strategy, **options):
        """ Make a tangential prediction

        Predicts the solution by incrementing lambda and all dofs with the
        increment of the last solution step.

        Parameters
        ----------
        strategy : str
            Strategy to scale the tangent vector.
            Avaiblable options:
            - 'lambda'
            - 'delta-lambda'
            - 'dof'
            - 'delta-dof'
            - 'arc-length'
        options
            value : float
                prescribed value according to the strategy
            dof : Object
                specifies the controlled dof for 'dof' and 'delta-dof' strategy
        """
        self.status = ModelStatus.prediction
        assembler = Assembler(self)

        # get tangent vector
        tangent = self.get_tangent_vector(assembler=assembler)

        # calculate scaling factor according to chosen strategy
        if strategy == 'lambda':
            prescribed_lam = options['value']

            delta_lambda = prescribed_lam - self.lam
            factor = delta_lambda

        elif strategy == 'delta-lambda':
            delta_lambda = options['value']

            factor = delta_lambda

        elif strategy == 'dof':
            dof = options['dof']
            value_prescribed = options['value']

            value = self[dof].delta

            delta_prescribed = value_prescribed - value

            dof_index = assembler.index_of_dof(dof)
            delta_current = tangent[dof_index]

            factor = delta_prescribed / delta_current

        elif strategy == 'delta-dof':
            dof = options['dof']
            delta_dof_prescribed = options['value']

            dof_index = assembler.index_of_dof(dof)
            delta_dof_current = tangent[dof_index]

            factor = delta_dof_prescribed / delta_dof_current

        elif strategy == 'arc-length':
            previous_model = self.get_previous_model()

            if previous_model.get_previous_model() is not None:
                previous_increment = previous_model.get_increment_vector(assembler)

            if 'value' in options.keys():
                prescribed_length = options['value']
            elif previous_model.get_previous_model() is not None:
                prescribed_length = la.norm(previous_increment)
            else:
                prescribed_length = 0.0

            if prescribed_length == 0.0:
                print("WARNING: The length of the prescribed increment is 0.0!")

            current_length = la.norm(tangent)

            factor = prescribed_length / current_length

            # tangent should point in a similar direction as the last increment
            if previous_model.get_previous_model() is not None:
                if previous_increment @ tangent < 0:
                    factor = -factor

        else:
            raise ValueError('Invalid strategy for prediction: {}'
                             .format(strategy))

        # scale tangent vector
        tangent *= factor

        # update dofs at model
        for index, dof in enumerate(assembler.dofs):
            self[dof].delta += tangent[index]

        # update lambda at model
        self.lam += tangent[-1]

    def combine_prediction_with_eigenvector(self, beta):
        """Combine the prediciton with the first eigenvector

        Parameters
        ----------
        beta : float
            factor between -1.0 and 1.0 used for a linear combination of the
            prediction with the eigenvector

        Raises
        ------
        RuntimeError
            If the model is not in prediction status
        ValueError
            If the beta is not between -1.0 and 1.0
        """
        if self.status != ModelStatus.prediction:
            raise RuntimeError('Model is not a predictor. Cannot combine with eigenvector!')

        if beta < -1.0 or beta > 1.0:
            raise ValueError('beta needs to be between -1.0 and 1.0')

        previous_model = self.get_previous_model()
        if previous_model.first_eigenvector_model is None:
            print('WARNING: solving eigenvalue problem in order to do branch switching')
            previous_model.solve_eigenvalues()

        eigenvector_model = previous_model.first_eigenvector_model

        assembler = Assembler(self)

        u_prediction = self.get_delta_dof_vector(model_b=previous_model, assembler=assembler)

        prediction_length = la.norm(u_prediction)

        eigenvector = eigenvector_model.get_delta_dof_vector(assembler=assembler)

        # scale eigenvector to the length of the prediction
        eigenvector *= (1.0/(la.norm(eigenvector)/prediction_length))

        prediction = u_prediction * (1.0 - abs(beta)) + eigenvector * beta

        delta_prediction = prediction - u_prediction

        # lambda = 0 for the eigenvector. Note: TRUSS.xls uses the same value as for the last increment
        delta_lam = - self.get_lam_increment()

        # update dofs at model
        for index, dof in enumerate(assembler.free_dofs):
            self[dof].delta += delta_prediction[index]

        # update lambda at model
        self.lam += delta_lam

    def scale_prediction(self, factor):
        """scale the prediction with a factor

        Parameters
        ----------
        factor : float
            factor used to scale the prediction


        Raises
        ------
        RuntimeError
            If the model is not in prediction status
        RuntimeError
            If the model has no previous model
        """
        if self.status != ModelStatus.prediction:
            raise RuntimeError('Model is not a predictor. Can only scale predictor!')

        if factor == 1.0:
            return

        previous_model = self.get_previous_model()

        if previous_model is None:
            raise RuntimeError('Previous Model is None!')

        assembler = Assembler(self)

        delta_dof_vector = self.get_delta_dof_vector(previous_model, assembler=assembler)

        delta_lambda = self.lam - previous_model.lam

        delta_dof_vector *= (factor - 1.0)
        delta_lambda *= (factor - 1.0)

        for i, dof in enumerate(assembler.free_dofs):
            self[dof].delta += delta_dof_vector[i]

        self.lam += delta_lambda

    def get_delta_dof_vector(self, model_b=None, assembler=None):
        """gets the delta dof between this and a given model_b as a numpy array

        Parameters
        ----------
        model_b : Model
            Model that is used as reference for the delta dof calculation. If
            not given, the initial model is used as reference.
        assembler: Assembler
            Assembler is used to order the dofs in the vector. If not given, a
            new assembler is created


        Returns
        ------
        delta : np.ndarray
            vector with the delta of all dofs

        Raises
        ------
        RuntimeError
            If the model is not in prediction status
        RuntimeError
            If the model has no previous model
        """
        if model_b is None:
            model_b = self.get_initial_model()

        if assembler is None:
            assembler = Assembler(self)

        delta = np.zeros(assembler.dof_count)

        for index, dof in enumerate(assembler.free_dofs):
            delta[index] = self[dof].delta - model_b[dof].delta

        return delta

    def load_displacement_curve(self, dof, skip_iterations=True):
        history = self.get_model_history(skip_iterations)

        data = np.zeros([2, len(history)])

        for i, self in enumerate(history):
            data[0, i] = self[dof].delta
            data[1, i] = self.lam

        return data

    def _repr_html_(self):
        from nfem.visualization.notebook_animation import show_animation
        return show_animation(self).data
