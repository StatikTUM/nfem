import numpy as np
from nfem.assembler import Assembler
from nfem.model_status import ModelStatus
from nfem.path_following_method import ArcLengthControl, DisplacementControl, LoadControl
from nfem.newton_raphson import newton_raphson_solve
from numpy.linalg import solve as linear_solve


class SolutionInfo:
    def __init__(self, iterations):
        self.iterations = iterations


def linear_step(model):
    assembler = Assembler(model)

    dof_count = assembler.dof_count

    u = np.zeros(dof_count)

    for dof in assembler.dofs:
        index = assembler.index_of_dof(dof)
        u[index] = model[dof].delta

    k = np.zeros((dof_count, dof_count))
    f = np.zeros(dof_count)

    for i, dof in enumerate(assembler.dofs):
        f[i] += model[dof].external_force

    assembler.assemble_matrix(k, lambda element: element.calculate_elastic_stiffness_matrix())

    f *= model.load_factor

    try:
        u = linear_solve(k, f)
    except np.linalg.LinAlgError:
        raise RuntimeError('Stiffness matrix is singular')

    for index, dof in enumerate(assembler.dofs):
        model[dof].delta = u[index]

    model.status = ModelStatus.equilibrium


def load_control_step(model, tolerance=1e-5, max_iterations=100, **options):
    constraint = LoadControl(model)
    return nonlinear_step(constraint, model, tolerance, max_iterations, **options)


def displacement_control_step(model, dof, tolerance=1e-5, max_iterations=100, **options):
    constraint = DisplacementControl(model, dof)
    return nonlinear_step(constraint, model, tolerance, max_iterations, **options)


def arc_length_control_step(model, tolerance=1e-5, max_iterations=100, **options):
    constraint = ArcLengthControl(model)
    return nonlinear_step(constraint, model, tolerance, max_iterations, **options)


def nonlinear_step(constraint, model, tolerance=1e-5, max_iterations=100, **options):
    # initialize working matrices and functions for newton raphson
    assembler = Assembler(model)
    dof_count = assembler.dof_count

    def calculate_system(x):
        # create a duplicate of the current state before updating and insert it in the history
        duplicate = model.get_duplicate()
        duplicate._previous_model = model._previous_model
        model._previous_model = duplicate
        duplicate.status = model.status

        # update status flag
        model.status = ModelStatus.iteration

        # update actual coordinates
        for index, dof in enumerate(assembler.dofs):
            model[dof].delta = x[index]

        # update lambda
        model.load_factor = x[-1]

        # initialize with zeros
        k = np.zeros((dof_count, dof_count))
        external_f = np.zeros(dof_count)
        internal_f = np.zeros(dof_count)

        # assemble stiffness
        assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())

        # assemble force

        for i, dof in enumerate(assembler.dofs):
            external_f[i] += model[dof].external_force

        assembler.assemble_vector(internal_f, lambda element: element.calculate_internal_forces())

        # assemble left and right hand side for newton raphson
        lhs = np.zeros((dof_count + 1, dof_count + 1))
        rhs = np.zeros(dof_count + 1)

        # mechanical system
        lhs[:dof_count, :dof_count] = k
        lhs[:dof_count, -1] = -external_f
        rhs[:dof_count] = internal_f - model.load_factor * external_f

        # assemble contribution from constraint
        constraint.calculate_derivatives(model, lhs[-1, :])
        rhs[-1] = constraint.calculate_constraint(model)

        return lhs, rhs

    # prediction as vector for newton raphson
    x = np.zeros(dof_count + 1)
    for index, dof in enumerate(assembler.dofs):
        x[index] = model[dof].delta

    x[-1] = model.load_factor

    # solve newton raphson
    x, n_iter = newton_raphson_solve(calculate_system, x, max_iterations, tolerance)

    model.status = ModelStatus.equilibrium

    if options.get('solve_det_k', True):
        model.solve_det_k(assembler=assembler)

    if options.get('solve_attendant_eigenvalue', False):
        model.solve_eigenvalues(assembler=assembler)

    return SolutionInfo(iterations=n_iter)
