"""Three dimensional Node."""

from __future__ import annotations

from nfem.dof import Dof

import numpy as np
import numpy.typing as npt

from typing import Sequence

Vector = npt.NDArray[np.float64]


class Node:
    """Three dimensional Node.

    :id: Unique ID.
    :ref_x: ref X coordinate.
    :ref_y: ref Y coordinate.
    :ref_z: ref Z coordinate.
    :x: Actual X coordinate.
    :y: Actual Y coordinate.
    :z: Actual Z coordinate.
    :u: Displacement in x direction.
    :v: Displacement in y direction.
    :w: Displacement in z direction.
    """

    id: str
    _dof_x: Dof
    _dof_y: Dof
    _dof_z: Dof

    def __init__(self, id: str, x: float, y: float, z: float):
        """Create a new node.

        :id: Unique ID of the node.
        :x: Initial X coordinate of the node.
        :y: Initial Y coordinate of the node.
        :z: Initial Z coordinate of the node.
        """
        self.id = id
        self._dof_x = Dof(id=(id, 'u'), value=x)
        self._dof_y = Dof(id=(id, 'v'), value=y)
        self._dof_z = Dof(id=(id, 'w'), value=z)

    def dof(self, dof_type: str) -> Dof:
        """Get the degree of freedom for a specific direction."""
        if dof_type == 'u':
            return self._dof_x
        if dof_type == 'v':
            return self._dof_y
        if dof_type == 'w':
            return self._dof_z
        raise AttributeError(f'Node has no dof of type "{dof_type}"')

    # reference location

    @property
    def ref_x(self) -> float:
        """Get or set the x coordinate in the undeformed configuration."""
        return self._dof_x.ref_value

    @ref_x.setter
    def ref_x(self, value: float) -> None:
        self._dof_x.ref_value = value

    @property
    def ref_y(self) -> float:
        """Get or set the y coordinate in the undeformed configuration."""
        return self._dof_y.ref_value

    @ref_y.setter
    def ref_y(self, value: float) -> None:
        self._dof_y.ref_value = value

    @property
    def ref_z(self) -> float:
        """Get or set the z coordinate in the undeformed configuration."""
        return self._dof_z.ref_value

    @ref_z.setter
    def ref_z(self, value: float) -> None:
        self._dof_z.ref_value = value

    @property
    def ref_location(self) -> Vector:
        """Get or set the location in the undeformed configuration."""
        return np.array([self.ref_x, self.ref_y, self.ref_z])

    @ref_location.setter
    def ref_location(self, value: Sequence[float]):
        self.ref_x, self.ref_y, self.ref_z = value

    # actual location

    @property
    def x(self) -> float:
        """Get or set the x coordinate in the deformed configuration."""
        return self._dof_x.value

    @x.setter
    def x(self, value: float) -> None:
        self._dof_x.value = value

    @property
    def y(self) -> float:
        """Get or set the y coordinate in the deformed configuration."""
        return self._dof_y.value

    @y.setter
    def y(self, value: float) -> None:
        self._dof_y.value = value

    @property
    def z(self) -> float:
        """Get or set the z coordinate in the deformed configuration."""
        return self._dof_z.value

    @z.setter
    def z(self, value: float) -> None:
        self._dof_z.value = value

    @property
    def location(self) -> Vector:
        """Get or set the location in the deformed configuration."""
        return np.array([self.x, self.y, self.z])

    @location.setter
    def location(self, value: Sequence[float]):
        self.x, self.y, self.z = value

    # displacements

    @property
    def u(self) -> float:
        """Get or set the displacement in x direction."""
        return self._dof_x.delta

    @u.setter
    def u(self, value: float) -> None:
        self._dof_x.delta = value

    @property
    def v(self) -> float:
        """Get or set the displacement in y direction."""
        return self._dof_y.delta

    @v.setter
    def v(self, value: float) -> None:
        self._dof_y.delta = value

    @property
    def w(self) -> float:
        """Get or set the displacement in z direction."""
        return self._dof_z.delta

    @w.setter
    def w(self, value: float) -> None:
        self._dof_z.delta = value

    @property
    def displacement(self) -> Vector:
        """Get or set the displacement."""
        return np.array([self.u, self.v, self.w])

    @displacement.setter
    def displacement(self, value: Sequence[float]):
        self.u, self.v, self.w = value

    # external forces

    @property
    def fx(self) -> float:
        """Get or set the external force in x direction."""
        return self._dof_x.external_force

    @fx.setter
    def fx(self, value: float) -> None:
        self._dof_x.external_force = value

    @property
    def fy(self) -> float:
        """Get or set the external force in y direction."""
        return self._dof_y.external_force

    @fy.setter
    def fy(self, value: float) -> None:
        self._dof_y.external_force = value

    @property
    def fz(self) -> float:
        """Get or set the external force in z direction."""
        return self._dof_z.external_force

    @fz.setter
    def fz(self, value: float) -> None:
        self._dof_z.external_force = value

    @property
    def external_force(self) -> Vector:
        """Get or set the external force."""
        return np.array([self.fx, self.fy, self.fz])

    @external_force.setter
    def external_force(self, value: Sequence[float]):
        [self.fx, self.fy, self.fz] = value

    # residual forces

    @property
    def rx(self) -> float:
        """Get or set the residual force in x direction."""
        return self._dof_x.residual

    @rx.setter
    def rx(self, value: float) -> None:
        self._dof_x.residual = value

    @property
    def ry(self) -> float:
        """Get or set the residual force in y direction."""
        return self._dof_y.residual

    @ry.setter
    def ry(self, value: float) -> None:
        self._dof_y.residual = value

    @property
    def rz(self) -> float:
        """Get or set the residual force in z direction."""
        return self._dof_z.residual

    @rz.setter
    def rz(self, value: float) -> None:
        self._dof_z.residual = value

    @property
    def residual(self) -> Vector:
        """Get or set the residual force."""
        return np.array([self.rx, self.ry, self.rz])

    @residual.setter
    def residual(self, value: Sequence[float]):
        [self.rx, self.ry, self.rz] = value

    # supports

    @property
    def support_x(self) -> bool:
        """Get or set the support in x direction."""
        return not self._dof_x.is_active

    @support_x.setter
    def support_x(self, value: bool) -> None:
        self._dof_x.is_active = not value

    @property
    def support_y(self) -> bool:
        """Get or set the support in y direction."""
        return not self._dof_y.is_active

    @support_y.setter
    def support_y(self, value: bool) -> None:
        self._dof_y.is_active = not value

    @property
    def support_z(self) -> bool:
        """Get or set the support in z direction."""
        return not self._dof_z.is_active

    @support_z.setter
    def support_z(self, value: bool) -> None:
        self._dof_z.is_active = not value

    @property
    def support(self) -> str:
        """Get or set the supports."""
        result = ''
        if self.support_x:
            result += 'x'
        if self.support_y:
            result += 'y'
        if self.support_z:
            result += 'z'
        return result

    @support.setter
    def support(self, value: str) -> None:
        self.support_x = 'x' in value
        self.support_y = 'y' in value
        self.support_z = 'z' in value

    # visualization

    def draw(self, item):
        """Draw the node."""
        item.set_label_location(self.ref_location, self.location)

        item.add_support(
            location=self.ref_location,
            direction=self.support,
            layer=10,
            color='gray',
        )

        item.add_support(
            location=self.location,
            direction=self.support,
            layer=20,
            color='black',
        )

        item.add_point(
            location=self.ref_location,
            layer=10,
            color='gray',
        )

        item.add_point(
            location=self.location,
            layer=20,
            color='black',
        )

        d = self.location - self.ref_location

        item.add_result('Location undeformed', self.ref_location.tolist())
        item.add_result('Location', self.location.tolist())
        item.add_result('Displacement', d.tolist())
        item.add_result('Displacement X', float(d[0]))
        item.add_result('Displacement Y', float(d[1]))
        item.add_result('Displacement Z', float(d[2]))

        if np.linalg.norm(self.external_force) > 1e-8:
            direction = self.external_force
            direction /= np.linalg.norm(direction)

            item.add_arrow(
                location=self.ref_location - direction,
                direction=direction,
                layer=13,
                color='gray',
            )

            item.add_arrow(
                location=self.location - direction,
                direction=direction,
                layer=23,
                color='blue',
            )
