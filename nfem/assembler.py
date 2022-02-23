"""Auxiliary for assembling system vectors and system matrices."""

from __future__ import annotations

from nfem.dof import Dof
from nfem.element import Element
import nfem

import numpy as np
import numpy.typing as npt

from typing import Callable, Dict, Sequence, Tuple


EvalElement = Callable[[Element], npt.ArrayLike]


class Assembler:
    """Auxiliary for assembling system vectors and system matrices."""

    def __init__(self, model: nfem.Model):
        """Create a new Assembler.

        :model: Model of the system to be assembled.
        """
        dof_indices: Dict[Dof, int] = {}
        element_indices: Dict[Element, npt.NDArray[int]] = {}

        index = -1

        for element in model.elements:
            indices = np.empty(len(element.dofs), int)

            for i, dof in enumerate(element.dofs):
                dof_index = dof_indices.get(dof, None)

                if dof_index is None:
                    if dof.is_active:
                        index += 1
                        dof_index = index
                    else:
                        dof_index = index - len(dof_indices)

                    dof_indices[dof] = dof_index

                indices[i] = dof_index

            element_indices[element] = indices

        dofs = np.empty(len(dof_indices), object)

        for dof, i in dof_indices.items():
            dofs[i] = dof

        self.dofs: npt.NDArray[Dof] = dofs
        """List of all degrees of freedom including the locked ones."""

        self.dof_indices: Dict[Dof, int] = dof_indices
        """Provides the index for a given degree of freedom."""

        self.element_indices: Dict[Element, npt.NDArray] = element_indices
        """Indices of the degrees of freedom for a given element."""

        self.size: Tuple[int, int] = (index + 1, len(dofs))
        """Number of degrees of freedom (active and total)."""

    def assemble_vector(self, fn: EvalElement, out=None) -> npt.NDArray:
        """Assemble a system vector.

        :fn: Function to compute the local vector of each element.
        :out: Vector to save the result. (optional)
        :return: The assembled system vector.
        """
        if out is None:
            m = len(self.dofs)
            out = np.zeros(m, float)

        for element, indices in self.element_indices.items():
            local_vector = fn(element)
            out[indices] += local_vector

        return out

    def assemble_matrix(self, fn: EvalElement, out=None) -> npt.NDArray:
        """Assemble a system matrix.

        :fn: Function to compute the local matrix of each element.
        :out: matrix to save the result. (optional)
        :return: The assembled system matrix.
        """
        if out is None:
            m = len(self.dofs)
            out = np.zeros((m, m), float)

        for element, indices in self.element_indices.items():
            local_matrix = fn(element)
            out[np.ix_(indices, indices)] += local_matrix

        return out
