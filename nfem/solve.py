"""Strategies for solving linear and nonlinear FE problems."""

from __future__ import annotations

import nfem
from nfem.assembler import Assembler
from nfem.element import Element
from nfem.nonlinear_solution_data import NonlinearSolutionInfo
from nfem.model_status import ModelStatus

import numpy as np
import numpy.linalg as la

import io
from typing import Tuple


def solve_linear(model: nfem.Model):
    """Solve the linear FE problem."""
    assembler = Assembler(model)

    n, m = assembler.size

    r = np.zeros(m)
    k = np.zeros((m, m))

    # compute residual forces of the system

    for i in range(m):
        r[i] = -assembler.dofs[i].external_force

    r *= model.load_factor

    assembler.assemble_vector(_element_linear_r, out=r)

    # compute stiffness matrix of the system

    assembler.assemble_matrix(_element_linear_k, out=k)

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

    assembler.assemble_vector(_element_linear_r, out=r)

    # update residual forces

    for i in range(m):
        assembler.dofs[i].residual = r[i]

    model.status = ModelStatus.equilibrium

    return SolutionInfo(converged=True, iterations=1, residual_norm=0)


def solve_load_control(model: nfem.Model, tolerance: float = 1e-5,
                       max_iterations: int = 100):
    """Solve the nonlinear FE problem using load control."""
    assembler = Assembler(model)

    load_factor_hat = model.load_factor

    n, m = assembler.size

    r = np.empty(m)
    k = np.empty((m, m))

    rhs = np.zeros(n + 1)
    lhs = np.zeros((n + 1, n + 1))

    data = []

    iteration = 0

    while True:
        # compute residual forces of the system

        for i in range(m):
            r[i] = -assembler.dofs[i].external_force

        r *= model.load_factor

        assembler.assemble_vector(_element_r, out=r)

        # build right-hand-side of the equation system

        rhs[:n] = -r[:n]
        rhs[n] = model.load_factor - load_factor_hat

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
        assembler.assemble_matrix(_element_k, out=k)

        # build left-hand-side of the equation system

        lhs.fill(0)
        lhs[:n, :n] = k[:n, :n]
        lhs[n, n] = 1
        for i in range(n):
            lhs[i, n] = -assembler.dofs[i].external_force

        # solve linear equation system: lhs * delta = rhs

        try:
            delta = la.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError('Stiffness matrix is singular')

        # update model

        for i in range(n):
            assembler.dofs[i].value += delta[i]

        model.load_factor += delta[n]

        data.append([
            _format(model.load_factor),
            _format(rnorm),
            _format(la.norm(delta)),
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

    return NonlinearSolutionInfo(rnorm, ['λ', '|r|', '|du|'], data)


def solve_displacement_control(model: nfem.Model, dof: Tuple[str, str],
                               tolerance: float = 1e-5,
                               max_iterations: int = 100):
    """Solve the nonlinear FE problem using displacement control."""
    assembler = Assembler(model)

    dof_index = assembler.dof_indices[dof]
    d_hat = assembler.dofs[dof_index].delta

    n, m = assembler.size

    r = np.empty(m)
    k = np.empty((m, m))

    rhs = np.zeros(n + 1)
    lhs = np.zeros((n + 1, n + 1))

    data = []

    iteration = 0

    while True:
        # compute residual forces of the system

        for i in range(m):
            r[i] = -assembler.dofs[i].external_force
        r *= model.load_factor

        assembler.assemble_vector(_element_r, out=r)

        # build right-hand-side of the equation system

        rhs.fill(0)
        rhs[:n] = -r[:n]
        rhs[n] = model[dof].delta - d_hat

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
        assembler.assemble_matrix(_element_k, out=k)

        # build left-hand-side of the equation system

        lhs.fill(0)
        lhs[:n, :n] = k[:n, :n]
        lhs[n, dof_index] = 1
        for i in range(n):
            lhs[i, n] = -assembler.dofs[i].external_force

        # solve linear equation system: lhs * delta = rhs

        try:
            delta = la.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError('Stiffness matrix is singular')

        # update model

        for i in range(n):
            assembler.dofs[i].value += delta[i]

        model.load_factor += delta[n]

        data.append([
            _format(model.load_factor),
            _format(rnorm),
            _format(la.norm(delta)),
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

    return NonlinearSolutionInfo(rnorm, ['λ', '|r|', '|du|'], data)


def solve_arc_length_control(model: nfem.Model, tolerance: float = 1e-5,
                             max_iterations: int = 100):
    """Solve the nonlinear FE problem using arc-length control."""
    assembler = Assembler(model)

    previous_model = model.get_previous_model()

    def compute_squared_arc_length(model):
        squared_l = (model.load_factor - previous_model.load_factor)**2

        for node, previous_node in zip(model.nodes, previous_model.nodes):
            d = node.location - previous_node.location
            squared_l += d @ d

        return squared_l

    squared_l_hat = compute_squared_arc_length(model)

    n, m = assembler.size

    r = np.empty(m)
    k = np.empty((m, m))

    rhs = np.zeros(n + 1)
    lhs = np.zeros((n + 1, n + 1))

    data = []

    iteration = 0

    while True:
        # compute residual forces of the system

        for i in range(m):
            r[i] = -assembler.dofs[i].external_force
        r *= model.load_factor

        assembler.assemble_vector(_element_r, out=r)

        # build right-hand-side of the equation system

        rhs.fill(0)
        rhs[:n] = -r[:n]
        rhs[n] = -compute_squared_arc_length(model) + squared_l_hat

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
        assembler.assemble_matrix(_element_k, out=k)

        # build left-hand-side of the equation system

        lhs.fill(0)
        lhs[:n, :n] = k[:n, :n]
        for i in range(n):
            lhs[i, n] = -assembler.dofs[i].external_force

        for i, dof in enumerate(assembler.dofs[:n]):
            lhs[n, i] = 2 * (model[dof].delta - previous_model[dof].delta)

        lhs[n, n] = 2 * (model.load_factor - previous_model.load_factor)

        # solve linear equation system: lhs * delta = rhs

        try:
            delta = la.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            raise RuntimeError('Stiffness matrix is singular')

        # update model

        for i in range(n):
            assembler.dofs[i].value += delta[i]

        model.load_factor += delta[n]

        data.append([
            _format(model.load_factor),
            _format(rnorm),
            _format(la.norm(delta)),
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

    return NonlinearSolutionInfo(rnorm, ['λ', '|r|', '|du|'], data)


def _element_linear_r(element: Element):
    return element.compute_linear_r()


def _element_linear_k(element: Element):
    return element.compute_linear_k()


def _element_r(element: Element):
    return element.compute_r()


def _element_k(element: Element):
    return element.compute_k()


def _format(value, digits=6):
    return f'{value:.6e}' if value is not None else ''


class SolutionInfo:
    """Container for linear solution info."""

    def __init__(self, converged: bool, iterations: int, residual_norm: float):
        """Create a new linear SolutionInfo."""
        self.converged = converged
        self.iterations = iterations
        self.residual_norm = residual_norm

    def __repr__(self) -> str:
        """Get a text representation of the object."""
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
