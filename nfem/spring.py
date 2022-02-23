"""Linear spring element."""

from __future__ import annotations

from nfem.dof import Dof
from nfem.node import Node

import numpy as np
import numpy.typing as npt

from typing import Sequence


class Spring:
    """Linear spring element."""

    def __init__(self, id: str, node: Node, kx: float = 0.0, ky: float = 0.0,
                 kz: float = 0.0):
        """Create a new spring element.

        :id: Unique ID.
        :node: Adjacent node.
        :kx: Stiffness in x direction.
        :ky: Stiffness in y direction.
        :kz: Stiffness in z direction.
        """
        self.id = id
        self.node = node
        self.kx = kx
        self.ky = ky
        self.kz = kz

    @property
    def dofs(self) -> Sequence[Dof]:
        """Get the degrees of freedom."""
        node = self.node
        return [node._dof_x, node._dof_y, node._dof_z]

    # linear analysis

    def compute_linear_r(self) -> npt.NDArray[float]:
        """Compute the linear residual force vector of the element."""
        return np.array([
            self.kx * self.node.u,
            self.ky * self.node.v,
            self.kz * self.node.w,
        ], float)

    def compute_linear_k(self) -> npt.NDArray[float]:
        """Compute the linear stiffness matrix of the element."""
        return np.array([
            [self.kx, 0, 0],
            [0, self.ky, 0],
            [0, 0, self.kz],
        ], float)

    def compute_linear_kg(self) -> npt.NDArray[float]:
        """Compute the linear geometric stiffness matrix of the element."""
        return np.zeros((3, 3))

    # nonlinear analysis

    def compute_r(self):
        """Compute the nonlinear residual force vector of the element."""
        return self.compute_linear_r()

    def compute_k(self) -> npt.NDArray[float]:
        """Compute the nonlinear stiffness matrix of the element."""
        return self.compute_ke()

    def compute_ke(self) -> npt.NDArray[float]:
        """Compute the elastic stiffness matrix of the element."""
        return self.compute_linear_k()

    def compute_km(self) -> npt.NDArray[float]:
        """Compute the material stiffness matrix of the element."""
        return self.compute_linear_k()

    def compute_kg(self) -> npt.NDArray[float]:
        """Compute the geometric stiffness matrix of the element."""
        return np.zeros((3, 3))

    def compute_kd(self) -> npt.NDArray[float]:
        """Compute the initial-displacement stiffness matrix of the element."""
        return np.zeros((3, 3))

    # visualization

    def draw(self, canvas):
        """Draw the spring."""
        location = self.node.location
        if self.kx != 0:
            canvas.spring(location, 'x')
        if self.ky != 0:
            canvas.spring(location, 'y')
        if self.kz != 0:
            canvas.spring(location, 'z')
