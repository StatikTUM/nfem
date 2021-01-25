"""This module contains the Model class.

Authors: Thomas Oberbichler, Armin Geiser
"""

from copy import deepcopy
from typing import List, Optional, Type

import numpy as np
import numpy.linalg as la

from scipy.linalg import eig

from nfem.dof import Dof
from nfem.key_collection import KeyCollection
from nfem.model_status import ModelStatus
from nfem.node import Node
from nfem.truss import Truss
from nfem.spring import Spring

from nfem.assembler import Assembler

from nfem import solve


class Model:
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
    load_factor : float
        load factor
    previous_model : Model
        Previous state of this model
    """
    nodes: KeyCollection[str, Node]

    def __init__(self, name=None):
        """Create a new model.

        Parameters
        ----------
        name : str
            Name of the model.
        """

        self.name = name
        self.status = ModelStatus.initial
        self.nodes = KeyCollection()
        self.elements = KeyCollection()
        self.load_factor = 0.0
        self._previous_model = None
        self.det_k = None
        self.first_eigenvalue = None
        self.first_eigenvector_model = None

    # === modeling

    def add_node(self, id: str, x: float, y: float, z: float, support: str = '', fx: float = 0.0, fy: float = 0.0, fz: float = 0.0):
        """Add a three dimensional node to the model.

        Parameters
        ----------
        id: str
            Unique ID of the node.
        x: float
            X coordinate.
        y: float
            Y coordinate.
        z: float
            Z coordinate.
        support: str, optional
            Directions in which the displacements are fixed.
        fx: float, optional
            External force in x direction.
        fy: float, optional
            External force in y direction.
        fz: float, optional
            External force in z direction.
        """
        if not isinstance(id, str):
            raise TypeError('The node id is not a text string')

        if id in self.nodes:
            raise KeyError('The model already contains a node with id {}'.format(id))

        node = Node(id, x, y, 0 if z is None else z)

        self.nodes._add(node)

        if 'x' in support:
            node.dof('u').is_active = False
        if 'y' in support:
            node.dof('v').is_active = False
        if 'z' in support:
            node.dof('w').is_active = False

        node.dof('u').external_force = fx
        node.dof('v').external_force = fy
        node.dof('w').external_force = fz

    def add_truss(self, id: str, node_a: str, node_b: str, youngs_modulus: float, area: float, prestress: float = 0.0, tensile_strength: Optional[float] = None, compressive_strength: Optional[float] = None):
        """Add a three dimensional truss element to the model.

        Parameters
        ----------
        id : str
            Unique ID of the element.
        node_a : str
            ID of the first node.
        node_b : str
            ID of the second node.
        youngs_modulus : float
            Youngs modulus of the material for the truss.
        area : float
            Area of the cross section for the truss.
        tensile_strength : Optional[float]
            Tensile strength of the truss.
        compressive_strength : Optional[float]
            Compressive strength of the truss.

        Returns
        ----------
        Truss
            The new Truss

        Examples
        --------
        Add a truss element from node `A` to node `B`:

        >>> model.add_truss(node_a='A', node_a='B', youngs_modulus=20, area=1)
        """
        if not isinstance(id, str):
            raise TypeError('The element id is not a text string')

        if not isinstance(node_a, str):
            raise TypeError('The id of node_a is not a text string')

        if not isinstance(node_b, str):
            raise TypeError('The id of node_b is not a text string')

        if id in self.elements:
            raise KeyError('The model already contains an element with id {}'.format(id))

        if node_a not in self.nodes:
            raise KeyError('The model does not contain a node with id {}'.format(node_a))

        if node_b not in self.nodes:
            raise KeyError('The model does not contain a node with id {}'.format(node_b))

        element = Truss(id, self.nodes[node_a], self.nodes[node_b], youngs_modulus, area, prestress, tensile_strength, compressive_strength)

        self.elements._add(element)

    def add_spring(self, id: str, node: str, kx: float = 0.0, ky: float = 0.0, kz: float = 0.0):
        if not isinstance(id, str):
            raise TypeError('The element id is not a text string')

        if not isinstance(node, str):
            raise TypeError('The id of node is not a text string')

        if id in self.elements:
            raise KeyError('The model already contains an element with id {}'.format(id))

        if node not in self.nodes:
            raise KeyError('The model does not contain a node with id {}'.format(node))

        element = Spring(id, self.nodes[node], kx, ky, kz)

        self.elements._add(element)

    def add_element(self, element_type: Type, id: str, nodes: List[Node], **properties):
        if not isinstance(id, str):
            raise TypeError('The element id is not a text string')

        if id in self.elements:
            raise KeyError('The model already contains an element with id {}'.format(id))

        node_list = []

        for node in nodes:
            if not isinstance(node, str):
                raise TypeError(f'The id "{node}" is not a text string')

            if node not in self.nodes:
                raise KeyError('The model does not contain a node with id {}'.format(node))

            node_list.append(self.nodes[node])

        element = element_type(id, node_list, **properties)

        self.elements._add(element)

    # === degree of freedoms

    def __getitem__(self, key):
        if isinstance(key, Dof):
            node_key, dof_type = key.id
        else:
            node_key, dof_type = key
        return self.nodes[node_key].dof(dof_type)

    @property
    def dofs(self):
        return Assembler(self).dofs

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

        current_value = self.load_factor
        previous_value = self.get_previous_model().load_factor

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

    def perform_linear_solution_step(self, info=False):
        """Performs a linear solution step on the model.
            It uses the current load factor.
            The results are stored at the dofs and used to update the current
            coordinates of the nodes.
        """

        if info:
            print("Start linear solution step...")
            print("lambda : {}".format(self.load_factor))
            print()

        solve.linear_step(self)

    def perform_load_control_step(self, tolerance=1e-5, max_iterations=100, info=False, **options):
        solution_info = solve.load_control_step(self, tolerance, max_iterations, **options)
        if info:
            print(f'Load-Control with λ = {self.load_factor}')
            solution_info.show()
            print()

    def perform_displacement_control_step(self, dof, tolerance=1e-5, max_iterations=100, info=False, **options):
        solution_info = solve.displacement_control_step(self, dof, **options)
        if info:
            print(f'Displacement-Control with {dof[1]} at node {dof[0]} = {self[dof].delta}')
            solution_info.show()
            print()

    def perform_arc_length_control_step(self, tolerance=1e-5, max_iterations=100, info=False, **options):
        solution_info = solve.arc_length_control_step(self, **options)
        if info:
            print(f'Arc-Length-Control with length = {solution_info.constraint.squared_l_hat**0.5}')
            solution_info.show()
            print()

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

        if options.get('info', False):
            print("\n=================================")
            print("Start non linear solution step...")

        if strategy == 'load-control':
            info = solve.load_control_step(self, tolerance, max_iterations, **options)
        elif strategy == 'displacement-control':
            dof = options.pop('dof')
            info = solve.displacement_control_step(self, dof, tolerance, max_iterations, **options)
        elif strategy == 'arc-length-control':
            info = solve.arc_length_control_step(self, tolerance, max_iterations, **options)
        else:
            raise ValueError('Invalid path following strategy:' + strategy)

        if options.get('info', False):
            print(f'Residual norm: {info.residual_norm}.')
            print(f'Solution found after {info.iterations} iteration steps.')
            print()

    def get_stiffness(self, mode='comp'):
        assembler = Assembler(self)
        k = np.zeros((assembler.dof_count, assembler.dof_count))

        if mode == 'comp':
            assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())
        elif mode == 'elas':
            assembler.assemble_matrix(k, lambda element: element.calculate_elastic_stiffness_matrix())
        elif mode == 'disp':
            assembler.assemble_matrix(k, lambda element: element.calculate_initial_displacement_stiffness_matrix())
        elif mode == 'geom':
            assembler.assemble_matrix(k, lambda element: element.calculate_geometric_stiffness_matrix())
        else:
            raise ValueError('mode')

        return k

    def solve_det_k(self, k=None, assembler=None):
        """Solves the determinant of k

        Parameters
        ----------
        k : numpy.ndarray (optional)
            stiffness matrix can be directly passed.
        assembler : Object (optional)
            assembler can be passed to speed up if k is not given
        """
        solve.solve_det_k(self)
        print(f'Det(K): {self.det_k}')

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

        # assemble matrices
        k_e = np.zeros((dof_count, dof_count))
        k_g = np.zeros((dof_count, dof_count))
        print("=================================")
        print('Linearized prebuckling (LPB) analysis ...')
        assembler.assemble_matrix(k_e, lambda element: element.calculate_elastic_stiffness_matrix())
        assembler.assemble_matrix(k_g, lambda element: element.calculate_geometric_stiffness_matrix(linear=True))

        # solve eigenvalue problem
        eigvals, eigvecs = eig(k_e, -k_g)

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
        print('First linear eigenvalue * lambda: {}'.format(eigvals[0] * self.load_factor))  # this is printed in TRUSS
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
        model.load_factor = None

        for index, dof in enumerate(assembler.dofs):
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

        # assemble matrices
        k_m = np.zeros((dof_count, dof_count))
        k_g = np.zeros((dof_count, dof_count))
        print("=================================")
        print('Attendant eigenvalue analysis ...')
        assembler.assemble_matrix(k_m, lambda element: element.calculate_material_stiffness_matrix())
        assembler.assemble_matrix(k_g, lambda element: element.calculate_geometric_stiffness_matrix())

        # solve eigenvalue problem
        eigvals, eigvecs = eig(k_m, -k_g)

        # extract real parts of eigenvalues
        eigvals = np.array([x.real for x in eigvals])

        # sort eigenvalues and vectors
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # find index of closest eigenvalue to 1 (we could store all but that seems like an overkill)
        idx = (np.abs(eigvals - 1.0)).argmin()

        print('Closest eigenvalue: {}'.format(eigvals[idx]))
        print('Closest eigenvalue * lambda: {}'.format(eigvals[idx] * self.load_factor))  # this is printed in TRUSS
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
        model.load_factor = None

        for index, dof in enumerate(assembler.dofs):
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

        tangent = np.zeros(dof_count + 1)

        v = tangent[:-1]

        k = np.zeros((dof_count, dof_count))
        external_f = np.zeros(dof_count)

        # assemble stiffness
        assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

        # assemble force

        for i, dof in enumerate(assembler.dofs):
            external_f[i] += self[dof].external_force

        try:
            v[:dof_count] = la.solve(k, external_f)
        except np.linalg.LinAlgError:
            raise RuntimeError('Stiffness matrix is singular')

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
        self.load_factor = value

    def predict_load_increment(self, value):
        """Predicts the solution by incrementing lambda

        Parameters
        ----------
        value : float
            Value that is used to increment the load factor lambda.
        """
        self.status = ModelStatus.prediction
        self.load_factor += value

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
        self.load_factor += last_increment[-1]

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

            delta_lambda = prescribed_lam - self.load_factor
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
        self.load_factor += tangent[-1]

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
        for index, dof in enumerate(assembler.dofs):
            self[dof].delta += delta_prediction[index]

        # update lambda at model
        self.load_factor += delta_lam

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

        delta_lambda = self.load_factor - previous_model.load_factor

        delta_dof_vector *= (factor - 1.0)
        delta_lambda *= (factor - 1.0)

        for i, dof in enumerate(assembler.dofs):
            self[dof].delta += delta_dof_vector[i]

        self.load_factor += delta_lambda

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

        for index, dof in enumerate(assembler.dofs):
            delta[index] = self[dof].delta - model_b[dof].delta

        return delta

    def load_displacement_curve(self, dof, skip_iterations=True):
        history = self.get_model_history(skip_iterations)

        data = np.zeros([2, len(history)])

        for i, self in enumerate(history):
            data[0, i] = self[dof].delta
            data[1, i] = self.load_factor

        return data

    def _repr_html_(self):
        from nfem.visualization.canvas_3d import Canvas3D

        canvas = Canvas3D(height=600)

        return canvas.html(600, self).data

    def show(self, height=600, timestep=0):
        from nfem.visualization.canvas_3d import Canvas3D

        canvas = Canvas3D(height=height)

        canvas.show(height, self)
