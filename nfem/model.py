"""Model of a nonlinear finite element problem."""

from __future__ import annotations

from nfem import solve
from nfem.assembler import Assembler
from nfem.dof import Dof
from nfem.element import Element
from nfem.key_collection import KeyCollection
from nfem.model_status import ModelStatus
from nfem.node import Node
from nfem.spring import Spring
from nfem.truss import Truss

import numpy as np
import numpy.linalg as la
import numpy.typing as npt

from scipy.linalg import eig

from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Type, Union

DofID = Union[str, Dof, Tuple[str, str]]

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]


class Model:
    """Model of a nonlinear finite element problem."""

    def __init__(self, name: str = None):
        """Create a new model.

        :name: Name of the model.
        """
        self.name: str = name
        self.status: ModelStatus = ModelStatus.initial
        self.nodes: KeyCollection[str, Node] = KeyCollection()
        self.elements: KeyCollection[str, Element] = KeyCollection()
        self.load_factor: float = 0.0
        self._previous_model: Model = None
        self.det_k: Optional[float] = None
        self.first_eigenvalue: Optional[float] = None
        self.first_eigenvector_model: Optional[Vector] = None

    # === modeling

    def add_node(self, id: str, x: float, y: float, z: float,
                 support: str = '', fx: float = 0.0, fy: float = 0.0,
                 fz: float = 0.0) -> None:
        """Add a three dimensional node to the model.

        Parameters
        ----------
        @id: Unique ID of the node.
        @x: X coordinate.
        @y: Y coordinate.
        @z: Z coordinate.
        @support: Directions with supports.
        @fx: External force in x direction.
        @fy: External force in y direction.
        @fz: External force in z direction.
        """
        if not isinstance(id, str):
            raise TypeError('The node id is not a text string')

        if id in self.nodes:
            raise KeyError(f'The model already contains a node with id {id}')

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

    def add_truss(self, id: str, node_a: str, node_b: str,
                  youngs_modulus: float, area: float, prestress: float = 0.0,
                  tensile_strength: Optional[float] = None,
                  compressive_strength: Optional[float] = None) -> None:
        """Add a three dimensional truss element to the model.

        @id: Unique ID of the element.
        @node_a: ID of the first node.
        @node_b: ID of the second node.
        @youngs_modulus: Youngs modulus of the material for the truss.
        @area: Area of the cross section for the truss.
        @tensile_strength: Tensile strength of the truss.
        @compressive_strength: Compressive strength of the truss.

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
            raise KeyError('The model already contains an element with id ' +
                           f'"{id}"')

        if node_a not in self.nodes:
            raise KeyError('The model does not contain a node with id ' +
                           f'"{node_a}"')

        if node_b not in self.nodes:
            raise KeyError('The model does not contain a node with id ' +
                           f'"{node_b}"')

        element = Truss(id, self.nodes[node_a], self.nodes[node_b],
                        youngs_modulus, area, prestress)
        element.tensile_strength = tensile_strength
        element.compressive_strength = compressive_strength

        self.elements._add(element)

    def add_spring(self, id: str, node: str, kx: float = 0.0, ky: float = 0.0,
                   kz: float = 0.0) -> None:
        """Add a spring element.

        @id: Unique ID.
        @node: Adjacent node.
        @kx: Stiffness in x direction.
        @ky: Stiffness in y direction.
        @kz: Stiffness in z direction.
        """
        if not isinstance(id, str):
            raise TypeError('The element id is not a text string')

        if not isinstance(node, str):
            raise TypeError('The id of node is not a text string')

        if id in self.elements:
            raise KeyError('The model already contains an element with id ' +
                           f'"{id}"')

        if node not in self.nodes:
            raise KeyError('The model does not contain a node with id ' +
                           f'"{node}"')

        element = Spring(id, self.nodes[node], kx, ky, kz)

        self.elements._add(element)

    def add_element(self, element_type: Type, id: str, nodes: List[Node],
                    *args, **kwargs) -> None:
        """Add an element (Advanced!).

        @element_type: Factory for the new element.
        @id: Unique ID.
        @nodes: Adjacent Nodes.
        @args: Additional arguments passed to the factory.
        @kwargs: Additional arguments passed to the factory.
        """
        if not isinstance(id, str):
            raise TypeError('The element id is not a text string')

        if id in self.elements:
            raise KeyError('The model already contains an element with id ' +
                           f'"{id}"')

        node_list = []

        for node in nodes:
            if not isinstance(node, str):
                raise TypeError(f'The id "{node}" is not a text string')

            if node not in self.nodes:
                raise KeyError('The model does not contain a node with id ' +
                               f'"{node}"')

            node_list.append(self.nodes[node])

        element = element_type(id, node_list, *args, **kwargs)

        self.elements._add(element)

    # === degree of freedoms

    def __getitem__(self, key) -> Dof:
        """Get a degree of freedom."""
        if isinstance(key, str):
            return self.dof(key)
        elif isinstance(key, Dof):
            node_key, dof_type = key.id
        else:
            node_key, dof_type = key
        return self.nodes[node_key].dof(dof_type)

    @property
    def dofs(self) -> Sequence[Dof]:
        """Get all degrees of freedom."""
        assembler = Assembler(self)

        n, _ = assembler.size

        return assembler.dofs[:n]

    def dof(self, id: str) -> Dof:
        """Get a degree of freedom."""
        node, direction = id.rsplit('.', maxsplit=1)
        return self.nodes[node].dof(direction)

    # === r and k

    def compute_linear_r(self) -> Vector:
        """Compute the linear residual force vector of the element."""
        assembler = Assembler(self)

        n, m = assembler.size

        r = np.empty(m)

        for i, dof in enumerate(assembler.dofs):
            r[i] = -dof.external_force

        r[:] *= self.load_factor

        def compute_local(element: Element) -> Vector:
            return element.compute_linear_r()

        r = assembler.assemble_vector(compute_local)

        return r[:n]

    def compute_linear_k(self) -> Matrix:
        """Compute the linear stiffness matrix of the element."""
        assembler = Assembler(self)

        n, _ = assembler.size

        def compute_local(element: Element) -> Vector:
            return element.compute_linear_k()

        k = assembler.assemble_vector(compute_local)

        return k[:n, :n]

    def compute_linear_kg(self) -> Matrix:
        """Compute the linear geometric stiffness matrix of the element."""
        assembler = Assembler(self)

        n, _ = assembler.size

        def compute_local(element: Element) -> Vector:
            return element.compute_linear_kg()

        k = assembler.assemble_matrix(compute_local)

        return k[:n, :n]

    def compute_r(self) -> Vector:
        """Compute the nonlinear residual force vector of the element."""
        assembler = Assembler(self)

        n, m = assembler.size

        r = np.empty(m)

        for i, dof in enumerate(assembler.dofs):
            r[i] = -dof.external_force

        r[:] *= self.load_factor

        def compute_local(element: Element) -> Vector:
            return element.compute_r()

        assembler.assemble_vector(compute_local, out=r)

        return r[:n]

    def compute_k(self) -> Matrix:
        """Compute the nonlinear stiffness matrix of the element."""
        assembler = Assembler(self)

        n, _ = assembler.size

        def compute_local(element: Element) -> Vector:
            return element.compute_k()

        k = assembler.assemble_matrix(compute_local)

        return k[:n, :n]

    def compute_ke(self) -> Matrix:
        """Compute the elastic stiffness matrix of the element."""
        assembler = Assembler(self)

        n, _ = assembler.size

        def compute_local(element: Element) -> Vector:
            return element.compute_ke()

        k = assembler.assemble_matrix(compute_local)

        return k[:n, :n]

    def compute_km(self) -> Matrix:
        """Compute the material stiffness matrix of the element."""
        assembler = Assembler(self)

        n, _ = assembler.size

        def compute_local(element: Element) -> Vector:
            return element.compute_km()

        k = assembler.assemble_matrix(compute_local)

        return k[:n, :n]

    def compute_kg(self) -> Matrix:
        """Compute the geometric stiffness matrix of the element."""
        assembler = Assembler(self)

        n, _ = assembler.size

        def compute_local(element: Element) -> Vector:
            return element.compute_kg()

        k = assembler.assemble_matrix(compute_local)

        return k[:n, :n]

    def compute_kd(self) -> Matrix:
        """Compute the initial-displacement stiffness matrix of the element."""
        assembler = Assembler(self)

        n, _ = assembler.size

        def compute_local(element: Element) -> Vector:
            return element.compute_kd()

        k = assembler.assemble_matrix(compute_local)

        return k[:n, :n]

    def compute_det_k(self) -> float:
        """Compute the determinant of k."""
        k = self.compute_k()

        return la.det(k)

    # === increment

    def get_dof_increment(self, dof) -> float:
        """Get the increment of the dof during the last solution step.

        @dof: Dof that is asked.
        """
        if self.get_previous_model() is None:
            return 0.0

        previous_model = self.get_previous_model()

        current_value = self[dof].delta
        previous_value = previous_model[dof].delta

        return current_value - previous_value

    def get_lam_increment(self) -> float:
        """Get the increment of lambda during the last solution step."""
        if self.get_previous_model() is None:
            return 0.0

        current_value = self.load_factor
        previous_value = self.get_previous_model().load_factor

        return current_value - previous_value

    def get_increment_vector(self, assembler: Assembler = None) -> Vector:
        """Get the increment that resulted in the current position.

        @assembler: Initialized assembler.
        """
        if assembler is None:
            assembler = Assembler(self)

        n, _ = assembler.size

        increment = np.zeros(n + 1)

        if self.get_previous_model() is None:
            print('WARNING: Increment is zero because no previous model' +
                  ' exists!')
            return increment

        for index, dof in enumerate(assembler.dofs[:n]):
            increment[index] = self.get_dof_increment(dof)

        increment[-1] = self.get_lam_increment()

        return increment

    def get_increment_norm(self, assembler: Assembler = None) -> float:
        """Get the vector norm of the current increment.

        @assembler: Initialized assembler.
        """
        increment = self.get_increment_vector(assembler)
        return la.norm(increment)

    # === model history

    def get_previous_model(self, skip_iterations: bool = True) -> Model:
        """Get the previous model of the current model.

        @skip_iterations: True if iterations or predictions should be skipped.
        """
        if not skip_iterations:
            return self._previous_model

        # find the most previous model that is not an iteration or prediction
        previous_model = self._previous_model

        while (previous_model is not None and
               previous_model.status in [ModelStatus.duplicate,
                                         ModelStatus.prediction,
                                         ModelStatus.iteration]):
            previous_model = previous_model._previous_model

        return previous_model

    def get_initial_model(self) -> Model:
        """Get the initial model of this model."""
        current_model = self

        while current_model.get_previous_model() is not None:
            current_model = current_model.get_previous_model()

        return current_model

    def get_model_history(self, skip_iterations: bool = True) -> List[Model]:
        """Get a list of all previous models of this model.

        @skip_iterations: True if non converged models should be considered.
        """
        history = [self]

        current_model = self

        while current_model.get_previous_model(skip_iterations) is not None:
            current_model = current_model.get_previous_model(skip_iterations)

            history = [current_model] + history

        return history

    def get_duplicate(self, name: str = None, branch: bool = False) -> Model:
        r"""Get a duplicate of the model.

        @name: Name of the new model.
        @branch: True to create a new branch.


        Notes
        -----
        If `branch` is `False` the duplicate will be a successor of the
        current model:

            previous ----> current ----> duplicate

        If `branch` is `True` the duplicate will be a successor of the
        previous model:

            previous ----> current
                    \
                     \---> duplicate
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

    def new_timestep(self, name: str = None) -> Model:
        """Create a new timestep."""
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

    def perform_linear_solution_step(self, info: bool = False) -> None:
        """Perform a linear solution step on the model.

        It uses the current load factor.
        The results are stored at the dofs and used to update the current
        coordinates of the nodes.
        """
        if info:
            print("Start linear solution step...")
            print(f"lambda : {self.load_factor}")
            print()

        solve.solve_linear(self)

    def perform_load_control_step(self,
                                  tolerance: float = 1e-5,
                                  max_iterations: int = 500,
                                  info: bool = False,
                                  solve_det_k: bool = True,
                                  solve_attendant_eigenvalue: bool = False
                                  ) -> None:
        """Perform a solution step using load control."""
        solution = solve.solve_load_control(self, tolerance, max_iterations)

        if solve_det_k or solve_attendant_eigenvalue:
            assembler = Assembler(self)

            n, m = assembler.size

        if solve_det_k:
            k = np.zeros((m, m))
            assembler.assemble_matrix(lambda e: e.compute_k(), out=k)
            self.det_k = la.det(k[:n, :n])

        if solve_attendant_eigenvalue:
            self.solve_eigenvalues(assembler=assembler)

        if info:
            print(f'Load-Control with λ = {self.load_factor}')
            solution.show()
            print()

    def perform_displacement_control_step(self,
                                          dof: DofID,
                                          tolerance: float = 1e-5,
                                          max_iterations: int = 500,
                                          info: bool = False,
                                          solve_det_k: bool = True,
                                          solve_attendant_eigenvalue: bool
                                          = False
                                          ) -> None:
        """Perform a solution step using displacement control."""
        solution = solve.solve_displacement_control(self, dof, tolerance,
                                                    max_iterations)

        if solve_det_k or solve_attendant_eigenvalue:
            assembler = Assembler(self)

            n, m = assembler.size

        if solve_det_k:
            k = np.zeros((m, m))
            assembler.assemble_matrix(lambda e: e.compute_k(), out=k)
            self.det_k = la.det(k[:n, :n])

        if solve_attendant_eigenvalue:
            self.solve_eigenvalues(assembler=assembler)

        if info:
            print(f'Displacement-Control with {dof[1]} at node {dof[0]} =' +
                  f' {self[dof].delta}')
            solution.show()
            print()

    def perform_arc_length_control_step(self,
                                        tolerance: float = 1e-5,
                                        max_iterations: int = 500,
                                        info: bool = False,
                                        solve_det_k: bool = True,
                                        solve_attendant_eigenvalue: bool
                                        = False
                                        ) -> None:
        """Perform a solution step using arc-length control."""
        solution = solve.solve_arc_length_control(
            self, tolerance, max_iterations)

        if solve_det_k or solve_attendant_eigenvalue:
            assembler = Assembler(self)

            n, m = assembler.size

        if solve_det_k:
            k = np.zeros((m, m))
            assembler.assemble_matrix(lambda e: e.compute_k(), out=k)
            self.det_k = la.det(k[:n, :n])

        if solve_attendant_eigenvalue:
            self.solve_eigenvalues(assembler=assembler)

        if info:
            print('Arc-Length-Control with length = ' +
                  f'{solution.constraint.squared_l_hat**0.5}')
            solution.show()
            print()

    def get_stiffness(self, mode: str = 'comp') -> Matrix:
        """Get the stiffness matrix of the system."""
        if mode == 'comp':
            return self.compute_k()
        elif mode == 'elas':
            return self.compute_ke()
        elif mode == 'geom':
            return self.compute_kg()
        elif mode == 'disp':
            return self.compute_kd()

        raise ValueError('mode')

    def solve_det_k(self) -> None:
        """Compute, stores and prints the determinant of k."""
        self.det_k = self.compute_det_k()

        print(f'Det(K): {self.det_k}')

    def solve_linear_eigenvalues(self, assembler: Assembler = None) -> None:
        """Solve the linearized eigenvalue problem.

        @assembler: Initialized assembler.

        Eigenvalue problem:
        [ k_e + eigvals * k_g(linear strain) ] * eigvecs = 0

        Stores the first positive eigenvalue and vector
        """
        if assembler is None:
            assembler = Assembler(self)

        n, m = assembler.size

        # assemble matrices
        k_e = np.zeros((m, m))
        k_g = np.zeros((m, m))
        print("=================================")
        print('Linearized prebuckling (LPB) analysis ...')
        assembler.assemble_matrix(
            lambda element: element.compute_linear_k(), out=k_e)
        assembler.assemble_matrix(
            lambda element: element.compute_linear_kg(), out=k_g)

        # solve eigenvalue problem
        eigvals, eigvecs = eig(k_e[:n, :n], -k_g[:n, :n])

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

        print(f'First linear eigenvalue: {eigvals[0]}')
        # this is printed in TRUSS
        print('First linear eigenvalue * lambda:' +
              f' {eigvals[0] * self.load_factor}')
        if len(eigvecs[0]) < 10:
            print(f'First linear eigenvector: {eigvecs[0]}')

        self.first_eigenvalue = eigvals[0]

        # store eigenvector as model
        model = self.get_duplicate()
        model._previous_model = self
        model.status = ModelStatus.eigenvector
        model.det_k = None
        model.first_eigenvalue = None
        model.first_eigenvector_model = None
        model.load_factor = None

        for index, dof in enumerate(assembler.dofs[:n]):
            model[dof].delta = eigvecs[index][0]

        self.first_eigenvector_model = model

    def solve_eigenvalues(self, assembler: Assembler = None) -> None:
        """Solve the eigenvalue problem.

        @assembler: Initialized assembler.

        Eigenvalue problem:
        [ k_m + eigvals * k_g ] * eigvecs = 0

        Stores the closest (most critical) eigenvalue and vector
        """
        if assembler is None:
            assembler = Assembler(self)

        n, m = assembler.size

        # assemble matrices
        k_m = np.zeros((m, m))
        k_g = np.zeros((m, m))
        print("=================================")
        print('Attendant eigenvalue analysis ...')
        assembler.assemble_matrix(
            lambda element: element.compute_km(), out=k_m)
        assembler.assemble_matrix(
            lambda element: element.compute_kg(), out=k_g)

        # solve eigenvalue problem
        eigvals, eigvecs = eig(k_m[:n, :n], -k_g[:n, :n])

        # extract real parts of eigenvalues
        eigvals = np.array([x.real for x in eigvals])

        # sort eigenvalues and vectors
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # find index of closest eigenvalue to 1 (we could store all but that
        # seems like an overkill)
        idx = (np.abs(eigvals - 1.0)).argmin()

        print(f'Closest eigenvalue: {eigvals[idx]}')
        # this is printed in TRUSS
        print(
            f'Closest eigenvalue * lambda: {eigvals[idx] * self.load_factor}')
        if len(eigvecs[idx]) < 10:
            print('Closest eigenvector: {eigvecs[idx]}')

        self.first_eigenvalue = eigvals[idx]

        # store eigenvector as model
        model = self.get_duplicate()
        model._previous_model = self
        model.status = ModelStatus.eigenvector
        model.det_k = None
        model.first_eigenvalue = None
        model.first_eigenvector_model = None
        model.load_factor = None

        for index, dof in enumerate(assembler.dofs[:n]):
            model[dof].delta = eigvecs[index][idx]

        self.first_eigenvector_model = model

    def get_tangent_vector(self, assembler: Assembler = None) -> Vector:
        """Get the tangent vector.

        @tangent: Tangent vector t = [v, 1]
                  with v = d_u / d_lambda ... incremental velocity
        """
        if assembler is None:
            assembler = Assembler(self)

        n, m = assembler.size

        tangent = np.zeros(n + 1)

        k = np.zeros((m, m))
        external_f = np.zeros(n)

        # assemble stiffness
        assembler.assemble_matrix(lambda element: element.compute_k(), out=k)

        # assemble force

        for i in range(n):
            external_f[i] += assembler.dofs[i].external_force

        try:
            tangent[:n] = la.solve(k[:n, :n], external_f)
        except la.LinAlgError:
            raise RuntimeError('Stiffness matrix is singular')

        # load-factor = 1

        tangent[n] = 1

        return tangent

    # === prediction functions

    def predict_load_factor(self, value: float) -> None:
        """Predict the solution by predictor_method lambda.

        @value: Value for the new load factor lambda.
        """
        self.status = ModelStatus.prediction
        self.load_factor = value

    def predict_load_increment(self, value: float) -> None:
        """Predict the solution by incrementing lambda.

        @value: Value that is used to increment the load factor lambda.
        """
        self.status = ModelStatus.prediction
        self.load_factor += value

    def predict_dof_state(self, dof: DofID, value: float) -> None:
        """Predict the solution by predictor_method the dof.

        @dof: Dof that is prescribed.
        @value: Value that is used to prescribe the dof.
        """
        self.status = ModelStatus.prediction
        self[dof].delta = value

    def predict_dof_increment(self, dof: DofID, value: float) -> None:
        """Predict the solution by incrementing the dof.

        @dof: Dof that is incremented.
        @value: Value that is used to increment the dof.
        """
        self.status = ModelStatus.prediction
        self[dof].delta += value

    def predict_with_last_increment(self,
                                    value: Optional[float] = None
                                    ) -> None:
        """Predict the solution by incrementing lambda and all dofs.

        Uses the increment of the last solution step.

        @value: Length of the increment
        """
        self.status = ModelStatus.prediction
        if self.get_previous_model().get_previous_model() is None:
            raise RuntimeError('predict_with_last_increment can only be used' +
                               ' after the first step!')

        assembler = Assembler(self)

        previous_model = self.get_previous_model()

        last_increment = previous_model.get_increment_vector(assembler)

        length = la.norm(last_increment)

        if value is not None and length != 0.0:
            last_increment *= value / length
            length *= value

        if length == 0.0:
            print("WARNING: The length of the prescribed increment is 0.0!")

        # update dofs at model
        for index, dof in enumerate(assembler.dofs):
            self[dof].delta += last_increment[index]

        # update lam at model
        self.load_factor += last_increment[-1]

    def predict_tangential(self, strategy, **options):
        """Make a tangential prediction.

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

        n, _ = assembler.size

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

            dof_index = assembler.dof_indices[dof]
            delta_current = tangent[dof_index]

            factor = delta_prescribed / delta_current

        elif strategy == 'delta-dof':
            dof = options['dof']
            delta_dof_prescribed = options['value']

            dof_index = assembler.dof_indices[dof]
            delta_dof_current = tangent[dof_index]

            factor = delta_dof_prescribed / delta_dof_current

        elif strategy == 'arc-length':
            previous_model = self.get_previous_model()

            if previous_model.get_previous_model() is not None:
                previous_increment = previous_model.get_increment_vector(
                    assembler)

            if 'value' in options.keys():
                prescribed_length = options['value']
            elif previous_model.get_previous_model() is not None:
                prescribed_length = la.norm(previous_increment)
            else:
                prescribed_length = 0.0

            if prescribed_length == 0.0:
                print("WARNING: Length of the prescribed increment is 0.0!")

            current_length = la.norm(tangent)

            factor = prescribed_length / current_length

            # tangent should point in a similar direction as the last increment
            if previous_model.get_previous_model() is not None:
                if previous_increment @ tangent < 0:
                    factor = -factor

        else:
            raise ValueError(f'Invalid strategy for prediction: {strategy}')

        # scale tangent vector
        tangent *= factor

        # update dofs at model
        for i, dof in enumerate(assembler.dofs[:n]):
            self[dof].delta += tangent[i]

        # update lambda at model
        self.load_factor += tangent[-1]

    def combine_prediction_with_eigenvector(self, beta):
        """Combine the prediciton with the first eigenvector.

        @beta: Factor between -1.0 and 1.0 used for a linear combination of the
               prediction with the eigenvector.

        :raises RuntimeError: If the model is not in prediction status.
        :raises ValueError: If the beta is not between -1.0 and 1.0.
        """
        if self.status != ModelStatus.prediction:
            raise RuntimeError('Model is not a predictor. Cannot combine' +
                               ' with eigenvector!')

        if beta < -1.0 or beta > 1.0:
            raise ValueError('beta needs to be between -1.0 and 1.0')

        previous_model = self.get_previous_model()
        if previous_model.first_eigenvector_model is None:
            print('WARNING: solving eigenvalue problem in order to do branch' +
                  ' switching')
            previous_model.solve_eigenvalues()

        eigenvector_model = previous_model.first_eigenvector_model

        assembler = Assembler(self)

        u_prediction = self.get_delta_dof_vector(previous_model, assembler)

        prediction_length = la.norm(u_prediction)

        eigenvector = eigenvector_model.get_delta_dof_vector(
            assembler=assembler)

        # scale eigenvector to the length of the prediction
        eigenvector *= (1.0/(la.norm(eigenvector)/prediction_length))

        prediction = u_prediction * (1.0 - abs(beta)) + eigenvector * beta

        delta_prediction = prediction - u_prediction

        # lambda = 0 for the eigenvector. Note: TRUSS.xls uses the same value
        # as for the last increment
        delta_lam = - self.get_lam_increment()

        # update dofs at model

        n, _ = assembler.size
        for index, dof in enumerate(assembler.dofs[:n]):
            self[dof].delta += delta_prediction[index]

        # update lambda at model
        self.load_factor += delta_lam

    def scale_prediction(self, factor):
        """Scale the prediction with a factor.

        @factor: Factor used to scale the prediction.

        @raises RuntimeError: If the model is not in prediction status
        @raises RuntimeError: If the model has no previous model
        """
        if self.status != ModelStatus.prediction:
            raise RuntimeError('Model is not a predictor. Can only scale' +
                               ' predictor!')

        if factor == 1.0:
            return

        previous_model = self.get_previous_model()

        if previous_model is None:
            raise RuntimeError('Previous Model is None!')

        assembler = Assembler(self)

        n, _ = assembler.size

        delta_dof_vector = self.get_delta_dof_vector(
            previous_model, assembler=assembler)

        delta_lambda = self.load_factor - previous_model.load_factor

        delta_dof_vector *= (factor - 1.0)
        delta_lambda *= (factor - 1.0)

        for i, dof in enumerate(assembler.dofs[:n]):
            self[dof].delta += delta_dof_vector[i]

        self.load_factor += delta_lambda

    def get_delta_dof_vector(self, model_b: Model = None,
                             assembler: Assembler = None) -> Vector:
        """Get the delta dof between this and a given model_b as a numpy array.

        @model_b: Model that is used as reference for the delta dof
            calculation. If not given, the initial model is used as reference.
        @assembler: Assembler is used to order the dofs in the vector. If not
            given, a new assembler is created.

        @raises RuntimeError: If the model is not in prediction status.
        @raises RuntimeError: If the model has no previous model.
        """
        if model_b is None:
            model_b = self.get_initial_model()

        if assembler is None:
            assembler = Assembler(self)

        n, _ = assembler.size

        delta = np.zeros(n)

        for index, dof in enumerate(assembler.dofs[:n]):
            delta[index] = self[dof].delta - model_b[dof].delta

        return delta

    def load_displacement_curve(self, dof: DofID,
                                skip_iterations: bool = True):
        """Get the load displacement curve for a specific degree of freedom."""
        history = self.get_model_history(skip_iterations)

        data = np.zeros([2, len(history)])

        for i, self in enumerate(history):
            data[0, i] = self[dof].delta
            data[1, i] = self.load_factor

        return data

    def _repr_html_(self) -> str:
        from nfem.canvas_3d import Canvas3D

        canvas = Canvas3D(height=600)

        return canvas.html(600, self)

    def html(self) -> str:
        from nfem.canvas_3d import Canvas3D

        canvas = Canvas3D(height=600)

        return canvas.raw_html(600, self)

    def show(self, height: int = 600, timestep: int = 0) -> None:
        """Show the model."""
        try:
            from nfem.canvas_3d import Canvas3D
            canvas = Canvas3D(height=height)
            canvas.show(height, self)
        except (ImportError, NameError):
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            from PyQt6.QtWidgets import QApplication

            import sys

            html = self.html()

            app = QApplication(sys.argv)

            view = QWebEngineView()
            view.setWindowTitle('nfem')
            view.setHtml(html)
            view.show()

            sys.exit(app.exec())
