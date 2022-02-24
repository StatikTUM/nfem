"""Nonlinear truss element."""

from __future__ import annotations

from nfem.dof import Dof
from nfem.node import Node

import numpy as np
import numpy.linalg as la
import numpy.typing as npt

from typing import Optional, Sequence

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

dg = np.array([
    [-1, 0, 0, 1, 0, 0],
    [0, -1, 0, 0, 1, 0],
    [0, 0, -1, 0, 0, 1],
])


class Truss:
    """Nonlinear truss element."""

    node_a: Node
    node_b: Node
    youngs_modulus: float
    area: float
    prestress: float
    tensile_strength: Optional[float]
    compressive_strength: Optional[float]

    def __init__(self, id: str, node_a: Node, node_b: Node,
                 youngs_modulus: float, area: float, prestress: float = 0.0):
        """Create a new truss element.

        :node_a: First node.
        :node_b: Second node.
        :youngs_modulus: Young's modulus of the material.
        :area: Area of the cross section.
        :prestress: Internal prestress.
        """
        self.id = id
        self.node_a = node_a
        self.node_b = node_b
        self.youngs_modulus = youngs_modulus
        self.area = area
        self.prestress = prestress
        self.tensile_strength: Optional[float] = None
        self.compressive_strength: Optional[float] = None

    @property
    def dofs(self) -> Sequence[Dof]:
        """Get the degrees of freedom."""
        node_a = self.node_a
        node_b = self.node_b

        return [
            node_a._dof_x, node_a._dof_y, node_a._dof_z,
            node_b._dof_x, node_b._dof_y, node_b._dof_z,
        ]

    @property
    def ref_length(self) -> float:
        """Get the length of the undeformed truss."""
        a = self.node_a.ref_location
        b = self.node_b.ref_location

        return la.norm(b - a)  # type: ignore

    @property
    def length(self) -> float:
        """Get the length of the deformed truss."""
        a = self.node_a.location
        b = self.node_b.location

        return la.norm(b - a)  # type: ignore

    # strain and stress

    def compute_epsilon_gl(self) -> float:
        """Get the Green-Lagrange strain."""
        # reference base vector
        A1 = self.node_b.ref_location - self.node_a.ref_location

        # actual base vector
        a1 = self.node_b.location - self.node_a.location

        return (a1 @ a1) / (2 * A1 @ A1) - 0.5

    def compute_epsilon_lin(self) -> float:
        """Get the linear strain."""
        # reference base vector
        A1 = self.node_b.ref_location - self.node_a.ref_location

        # actual base vector
        a1 = self.node_b.location - self.node_a.location

        return (a1 @ A1) / (A1 @ A1) - 1

    def compute_sigma_pk2(self) -> float:
        """Get the second Piola-Kirchoff stress."""
        eps = self.compute_epsilon_gl()

        return eps * self.youngs_modulus + self.prestress

    @property
    def normal_force(self) -> float:
        """Get normal force."""
        sigma_b = self.compute_sigma_pk2() * self.length / self.ref_length

        return sigma_b * self.area

    # linear analysis

    def compute_linear_r(self) -> Vector:
        """Compute the linear residual force vector of the element."""
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = a1 @ A1 / A11 - 1
        sig = eps * self.youngs_modulus + self.prestress

        dp = sig * self.area * L / A11 * A1

        return dp @ dg

    def compute_linear_k(self) -> Matrix:
        """Compute the linear stiffness matrix of the element."""
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        ddp = self.youngs_modulus * self.area / A11**2 * L * np.outer(A1, A1)

        return dg.T @ ddp @ dg

    def compute_linear_kg(self) -> Matrix:
        """Compute the linear geometric stiffness matrix of the element."""
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = a1 @ A1 / A11 - 1
        sig = eps * self.youngs_modulus + self.prestress

        ddp = sig * self.area / A11 * L * np.eye(3)

        return dg.T @ ddp @ dg

    # nonlinear analysis

    def compute_r(self) -> Vector:
        """Compute the nonlinear residual force vector of the element."""
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = 0.5 * (a1 @ a1 / A11 - 1)
        sig = eps * self.youngs_modulus + self.prestress

        dp = sig * self.area * L / A11 * a1

        return dp @ dg

    def compute_k(self) -> Matrix:
        """Compute the nonlinear stiffness matrix of the element."""
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = 0.5 * (a1 @ a1 / A11 - 1)
        sig = eps * self.youngs_modulus + self.prestress

        ddp = (
            self.youngs_modulus * self.area / A11**2 * L * np.outer(a1, a1) +
            sig * self.area / A11 * L * np.eye(3)
        )

        return dg.T @ ddp @ dg

    def compute_ke(self) -> Matrix:
        """Compute the elastic stiffness matrix of the element."""
        return self.compute_linear_k()

    def compute_km(self) -> Matrix:
        """Compute the material stiffness matrix of the element."""
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        ddp = self.youngs_modulus * self.area / A11**2 * L * np.outer(a1, a1)

        return dg.T @ ddp @ dg

    def compute_kg(self) -> Matrix:
        """Compute the geometric stiffness matrix of the element."""
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = 0.5 * (a1 @ a1 / A11 - 1)
        sig = eps * self.youngs_modulus + self.prestress

        ddp = sig * self.area / A11 * L * np.eye(3)

        return dg.T @ ddp @ dg

    def compute_kd(self) -> Matrix:
        """Compute the initial-displacement stiffness matrix of the element."""
        km = self.compute_km()
        ke = self.compute_linear_k()

        return km - ke

    # visualization

    def draw(self, item) -> None:
        """Draw the truss."""
        sigma = self.compute_sigma_pk2()
        color = 'black'
        eta = None

        if self.compute_sigma_pk2() > 1e-3:
            color = 'blue'
            if self.tensile_strength is not None:
                sigma_max = self.tensile_strength
                eta = sigma / sigma_max
        elif self.compute_sigma_pk2() < -1e-3:
            color = 'red'
            if self.compressive_strength is not None:
                sigma_max = -self.compressive_strength
                eta = sigma / sigma_max
        elif (self.tensile_strength is not None and
              self.compressive_strength is not None):
            eta = 0.0

        item.set_label_location(
            ref=0.5 * (self.node_a.ref_location + self.node_b.ref_location),
            act=0.5 * (self.node_a.location + self.node_b.location),
        )

        item.add_line(
            points=[
                self.node_a.ref_location,
                self.node_b.ref_location,
            ],
            layer=10,
            color='gray',
        )

        item.add_line(
            points=[
                self.node_a.location,
                self.node_b.location,
            ],
            layer=20,
            color=color,
        )

        item.add_result('Length undeformed', self.ref_length)
        item.add_result('Length', self.length)
        item.add_result('Engineering Strain', self.compute_epsilon_lin())
        item.add_result('Green-Lagrange Strain', self.compute_epsilon_gl())
        item.add_result('PK2 Stress', sigma)
        item.add_result('Normal Force', self.normal_force)

        if eta is not None:
            item.add_result('Degree of Utilization', eta)
