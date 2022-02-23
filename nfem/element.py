"""Base type of a finite element."""

from nfem.dof import Dof
from nfem.node import Node

import numpy.typing as npt

from typing import Sequence

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


class Element(Protocol):
    """Base type of a finite element."""

    id: str
    """The unique ID of the element."""

    dofs: Sequence[Dof]
    """A list containing the local degrees of freedom."""

    nodes: Sequence[Node]
    """A list containing the adjacent nodes."""

    def compute_linear_r(self) -> npt.NDArray:
        """Compute the linear residual force vector of the element."""

    def compute_linear_k(self) -> npt.NDArray:
        """Compute the linear stiffness matrix of the element."""

    def compute_linear_kg(self) -> npt.NDArray:
        """Compute the linear geometric stiffness matrix of the element."""

    def compute_r(self) -> npt.NDArray:
        """Compute the nonlinear residual force vector of the element."""

    def compute_k(self) -> npt.NDArray:
        """Compute the nonlinear stiffness matrix of the element."""

    def compute_ke(self) -> npt.NDArray:
        """Compute the elastic stiffness matrix of the element."""

    def compute_km(self) -> npt.NDArray:
        """Compute the material stiffness matrix of the element."""

    def compute_kg(self) -> npt.NDArray:
        """Compute the geometric stiffness matrix of the element."""

    def compute_kd(self) -> npt.NDArray:
        """Compute the initial-displacement stiffness matrix of the element."""
