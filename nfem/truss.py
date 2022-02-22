from __future__ import annotations

from nfem.node import Node

import numpy as np
import numpy.linalg as la

from typing import Optional


class Truss:
    """FIXME"""

    node_a: Node
    node_b: Node
    youngs_modulus: float
    area: float
    prestress: float
    tensile_strength: Optional[float]
    compressive_strength: Optional[float]

    def __init__(self, id: str, node_a: Node, node_b: Node, youngs_modulus: float, area: float, prestress: float = 0.0, tensile_strength: Optional[float] = None, compressive_strength: Optional[float] = None):
        """FIXME"""

        self.id = id
        self.node_a = node_a
        self.node_b = node_b
        self.youngs_modulus = youngs_modulus
        self.area = area
        self.prestress = prestress
        self.tensile_strength = tensile_strength
        self.compressive_strength = compressive_strength

    @property
    def dofs(self):
        """FIXME"""
        node_a = self.node_a
        node_b = self.node_b

        return [node_a._dof_x, node_a._dof_y, node_a._dof_z, node_b._dof_x, node_b._dof_y, node_b._dof_z]

    @property
    def ref_length(self) -> float:
        """Gets the length of the undeformed truss"""
        a = self.node_a.ref_location
        b = self.node_b.ref_location

        return la.norm(b - a)

    @property
    def length(self) -> float:
        """Gets the length of the deformed truss"""
        a = self.node_a.location
        b = self.node_b.location

        return la.norm(b - a)

    # strain and stress

    def compute_epsilon_gl(self):
        # reference base vector
        A1 = self.node_b.ref_location - self.node_a.ref_location

        # actual base vector
        a1 = self.node_b.location - self.node_a.location

        # green-lagrange strain
        epsilon_GL = (a1 @ a1 - A1 @ A1) / (2 * A1 @ A1)

        return epsilon_GL

    def compute_epsilon_lin(self):
        """FIXME"""

        # reference base vector
        A1 = self.node_b.ref_location - self.node_a.ref_location

        # actual base vector
        a1 = self.node_b.location - self.node_a.location

        L = self.ref_length

        # project actual on reference
        projected_l = a1 @ A1 / np.sqrt(A1 @ A1)

        e_lin = (projected_l - L) / L

        return e_lin

    def compute_sigma_pk2(self):
        eps = self.compute_epsilon_gl()
        E = self.youngs_modulus

        # stress:
        sigma = eps * E + self.prestress

        return sigma

    @property
    def normal_force(self) -> float:
        biot_stress = self.compute_sigma_pk2() * self.length / self.ref_length
        A = self.area
        F = biot_stress * A

        return F

    # linear analysis

    def compute_linear_r(self):
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = a1 @ A1 / A11 - 1
        sig = eps * self.youngs_modulus + self.prestress

        dp = sig * self.area * L / A11 * A1

        return dp @ [[-1, 0, 0, 1, 0, 0],
                     [0, -1, 0, 0, 1, 0],
                     [0, 0, -1, 0, 0, 1]]

    def compute_linear_k(self):
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        ddp = self.youngs_modulus * self.area / A11**2 * L * np.outer(A1, A1)

        dg = np.array([
            [-1, 0, 0, 1, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 1],
        ])

        return dg.T @ ddp @ dg

    def compute_linear_kg(self):
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = a1 @ A1 / A11 - 1
        sig = eps * self.youngs_modulus + self.prestress

        ddp = sig * self.area / A11 * L * np.eye(3)

        dg = np.array([
            [-1, 0, 0, 1, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 1],
        ])

        return dg.T @ ddp @ dg

    # nonlinear analysis

    def compute_r(self):
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = 0.5 * (a1 @ a1 / A11 - 1)
        sig = eps * self.youngs_modulus + self.prestress

        dp = sig * self.area * L / A11 * a1

        return dp @ [[-1, 0, 0, 1, 0, 0],
                     [0, -1, 0, 0, 1, 0],
                     [0, 0, -1, 0, 0, 1]]

    def compute_k(self):
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = 0.5 * (a1 @ a1 / A11 - 1)
        sig = eps * self.youngs_modulus + self.prestress

        ddp = self.youngs_modulus * self.area / A11**2 * L * np.outer(a1, a1) + sig * self.area / A11 * L * np.eye(3)

        dg = np.array([
            [-1, 0, 0, 1, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 1],
        ])

        return dg.T @ ddp @ dg

    def compute_ke(self):
        return self.compute_linear_k()

    def compute_km(self):
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        ddp = self.youngs_modulus * self.area / A11**2 * L * np.outer(a1, a1)

        dg = np.array([
            [-1, 0, 0, 1, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 1],
        ])

        return dg.T @ ddp @ dg

    def compute_kg(self):
        a1 = self.node_b.location - self.node_a.location
        A1 = self.node_b.ref_location - self.node_a.ref_location

        A11 = A1 @ A1
        L = np.sqrt(A11)

        eps = 0.5 * (a1 @ a1 / A11 - 1)
        sig = eps * self.youngs_modulus + self.prestress

        ddp = sig * self.area / A11 * L * np.eye(3)

        dg = np.array([
            [-1, 0, 0, 1, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 1],
        ])

        return dg.T @ ddp @ dg

    def compute_kd(self):
        km = self.compute_km()
        ke = self.compute_linear_k()

        return km - ke

    # visualization

    def draw(self, item):
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
        elif self.tensile_strength is not None and self.compressive_strength is not None:
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
