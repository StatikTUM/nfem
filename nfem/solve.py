import numpy as np
from nfem.assembler import Assembler
from nfem.nonlinear_solution_data import NonlinearSolutionInfo
from nfem.model_status import ModelStatus
from nfem.path_following_method import ArcLengthControl, DisplacementControl, LoadControl
from numpy.linalg import det, norm, solve as linear_solve
import io


def format(value, digits=6):
    return '{:.6e}'.format(value) if value is not None else ''


class SolutionInfo:
    def __init__(self, converged, iterations, residual_norm):
        self.converged = converged
        self.iterations = iterations
        self.residual_norm = residual_norm

    def __repr__(self):
        output = io.StringIO()
        if self.converged:
            print(f'System converged!', file=output)
        else:
            print(f'System not converged!', file=output)
        print(f'# Iterations  = {self.iterations}', file=output)
        print(f'Residual Norm = {self.residual_norm}', file=output)
        contents = output.getvalue()
        output.close()
        return contents


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

    return SolutionInfo(converged=True, iterations=1, residual_norm=0)


def newton_raphson_solve(calculate_system, x_initial, max_iterations=100, tolerance=1e-7, callback=None):
    x = x_initial
    residual_norm = None

    for iteration in range(1, max_iterations + 1):
        # calculate left and right hand side
        lhs, rhs = calculate_system(x)

        # calculate residual
        residual_norm = norm(rhs)

        # check convergence
        if residual_norm < tolerance:
            return residual_norm, iteration

        # compute delta_x
        try:
            delta_x = linear_solve(lhs, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError('Stiffness matrix is singular')

        # update x
        x -= delta_x

        if callback:
            callback(iteration, residual_norm, norm(delta_x))

    raise RuntimeError(f'Newthon-Raphson did not converge after {max_iterations} steps. Residual norm: {residual_norm}')


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

    data = []

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

    def callback(k, rnorm, xnorm):
        load_factor_str = format(model.load_factor)
        rnorm_str = format(rnorm)
        xnorm_str = format(xnorm)
        data.append([load_factor_str, rnorm_str, xnorm_str])

    # prediction as vector for newton raphson
    x = np.zeros(dof_count + 1)
    for index, dof in enumerate(assembler.dofs):
        x[index] = model[dof].delta

    x[-1] = model.load_factor

    # solve newton raphson
    residual_norm, iterations = newton_raphson_solve(calculate_system, x, max_iterations, tolerance, callback)

    callback(iterations, residual_norm, None)

    model.status = ModelStatus.equilibrium

    if options.get('solve_det_k', True):
        solve_det_k(model, assembler=assembler)

    if options.get('solve_attendant_eigenvalue', False):
        model.solve_eigenvalues(assembler=assembler)

    return NonlinearSolutionInfo(constraint, residual_norm, ['λ', '|r|', '|du|'], data)


def solve_det_k(model, k=None, assembler=None):
    if k is None:
        if assembler is None:
            assembler = Assembler(model)
        dof_count = assembler.dof_count
        k = np.zeros((dof_count, dof_count))
        assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())
    model.det_k = det(k)
