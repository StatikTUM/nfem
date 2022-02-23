from __future__ import annotations

import numpy as np
from nfem.assembler import Assembler
from nfem.nonlinear_solution_data import NonlinearSolutionInfo
from nfem.model_status import ModelStatus
from nfem.path_following_method import ArcLengthControl, DisplacementControl, LoadControl
from numpy.linalg import det, norm, solve as linear_solve
import numpy.linalg as la
import io


def element_linear_r(element):
    return element.compute_linear_r()


def element_linear_k(element):
    return element.compute_linear_k()


def element_r(element):
    return element.compute_r()


def element_k(element):
    return element.compute_k()


def solve_linear(model):
    assembler = Assembler(model)

    n, m = assembler.size

    r = np.zeros(m)
    k = np.zeros((m, m))

    # compute residual forces of the system

    for i in range(m):
        r[i] = -assembler.dofs[i].external_force

    r *= model.load_factor

    assembler.assemble_vector(element_linear_r, out=r)

    # compute stiffness matrix of the system

    assembler.assemble_matrix(element_linear_k, out=k)

    # build right-hand-side

    rhs = -r[:n]

    # build left-hand-side

    lhs = k[:n, :n]

    # Solve linear equation system: lhs * delta = rhs

    try:
        delta = la.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        raise RuntimeError('Stiffness matrix is singular')

    # update model

    for i in range(n):
        assembler.dofs[i].value += delta[i]

    # compute residuals

    for i in range(m):
        r[i] = -assembler.dofs[i].external_force

    r *= model.load_factor

    assembler.assemble_vector(element_linear_r, out=r)

    # update residual forces

    for i in range(m):
        assembler.dofs[i].residual = r[i]

    model.status = ModelStatus.equilibrium

    return SolutionInfo(converged=True, iterations=1, residual_norm=0)


def solve_load_control(model, tolerance: float = 1e-5,
                       max_iterations: int = 100, solve_det_k: bool = True,
                       solve_attendant_eigenvalue: bool = False):
    assembler = Assembler(model)

    n, m = assembler.size

    r = np.zeros(m)
    k = np.zeros((m, m))

    data = []

    iteration = 0

    while True:
        # compute residual forces of the system

        for i in range(m):
            r[i] = -assembler.dofs[i].external_force

        r *= model.load_factor

        assembler.assemble_vector(element_r, out=r)

        # build right-hand-side of the equation system

        rhs = np.zeros(n + 1)
        rhs[:n] = -r[:n]
        rhs[n] = 0

        # check residual criterion

        rnorm = la.norm(rhs)

        if rnorm < tolerance:
            break

        # check iteration criterion

        if iteration >= max_iterations:
            break

        # create a duplicate of the current state before updating and insert it
        # in the history

        duplicate = model.get_duplicate()
        duplicate._previous_model = model._previous_model
        model._previous_model = duplicate
        duplicate.status = model.status
        model.status = ModelStatus.iteration

        # compute stiffness matrix of the system

        k.fill(0)
        assembler.assemble_matrix(element_k, out=k)

        # build left-hand-side of the equation system

        lhs = np.zeros((n + 1, n + 1))
        lhs[:n, :n] = k[:n, :n]
        lhs[n, n] = 1

        # solve linear equation system: lhs * delta = rhs

        try:
            delta = la.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError('Stiffness matrix is singular')

        # update model

        for i in range(n):
            assembler.dofs[i].value += delta[i]

        data.append([
            format(model.load_factor),
            format(rnorm),
            format(la.norm(delta)),
        ])

        iteration += 1

    if iteration >= max_iterations:
        raise RuntimeError(
            f'Newthon-Raphson did not converge after {max_iterations} steps.' +
            f' Residual norm: {rnorm}'
        )

    # update residual forces

    for i in range(m):
        assembler.dofs[i].residual = r[i]

    model.status = ModelStatus.equilibrium

    if solve_det_k:
        k.fill(0)
        assembler.assemble_matrix(element_k, out=k)
        model.det_k = det(k[:n, :n])

    if solve_attendant_eigenvalue:
        model.solve_eigenvalues(assembler=assembler)

    return NonlinearSolutionInfo(rnorm, ['λ', '|r|', '|du|'], data)


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
            print('System converged!', file=output)
        else:
            print('System not converged!', file=output)
        print(f'# Iterations  = {self.iterations}', file=output)
        print(f'Residual Norm = {self.residual_norm}', file=output)
        contents = output.getvalue()
        output.close()
        return contents


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


def displacement_control_step(model, dof, tolerance=1e-5, max_iterations=100, **options):
    constraint = DisplacementControl(model, dof)
    return nonlinear_step(constraint, model, tolerance, max_iterations, **options)


def arc_length_control_step(model, tolerance=1e-5, max_iterations=100, **options):
    constraint = ArcLengthControl(model)
    return nonlinear_step(constraint, model, tolerance, max_iterations, **options)


def nonlinear_step(constraint, model, tolerance=1e-5, max_iterations=100, **options):
    # initialize working matrices and functions for newton raphson
    assembler = Assembler(model)

    n, m = assembler.size

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
        for index, dof in enumerate(assembler.dofs[:n]):
            model[dof].delta = x[index]

        # update lambda
        model.load_factor = x[-1]

        # initialize with zeros
        k = np.zeros((m, m))
        external_f = np.zeros(m)
        internal_f = np.zeros(m)

        # assemble stiffness
        assembler.assemble_matrix(lambda element: element.compute_k(), out=k)

        # assemble force

        for i, dof in enumerate(assembler.dofs[:n]):
            external_f[i] += model[dof].external_force

        assembler.assemble_vector(lambda element: element.compute_r(), out=internal_f)

        # assemble left and right hand side for newton raphson
        lhs = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)

        # mechanical system
        lhs[:n, :n] = k[:n, :n]
        lhs[:n, -1] = -external_f[:n]
        rhs[:n] = internal_f[:n] - model.load_factor * external_f[:n]

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
    x = np.zeros(n + 1)
    for index, dof in enumerate(assembler.dofs[:n]):
        x[index] = model[dof].delta

    x[-1] = model.load_factor

    # solve newton raphson
    residual_norm, iterations = newton_raphson_solve(calculate_system, x, max_iterations, tolerance, callback)

    callback(iterations, residual_norm, None)

    model.status = ModelStatus.equilibrium

    if options.get('solve_det_k', True):
        compute_det_k(model, assembler=assembler)

    if options.get('solve_attendant_eigenvalue', False):
        model.solve_eigenvalues(assembler=assembler)

    return NonlinearSolutionInfo(residual_norm, ['λ', '|r|', '|du|'], data)


def compute_det_k(model, k=None, assembler=None):
    if k is None:
        if assembler is None:
            assembler = Assembler(model)
        n, m = assembler.size

        k = np.zeros((m, m))
        assembler.assemble_matrix(lambda element: element.compute_k(), out=k)
    model.det_k = det(k[:n, :n])
