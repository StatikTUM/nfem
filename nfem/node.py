"""This module only contains the Node class.

Author: Thomas Oberbichler
"""

from typing import List
import numpy as np
from nfem.dof import Dof


class Node:
    """Three dimensional Node providing Dofs for displacements.

    Attributes
    ----------
    id : str
        Unique ID.
    ref_x : float
        ref X coordinate.
    ref_y : float
        ref Y coordinate.
    ref_z : float
        ref Z coordinate.
    x : float
        Actual X coordinate.
    y : float
        Actual Y coordinate.
    z : float
        Actual Z coordinate.
    u : float
        Displacement in x direction.
    v : float
        Displacement in y direction.
    w : float
        Displacement in z direction.
    """
    id: str
    _dof_x: Dof
    _dof_y: Dof
    _dof_z: Dof

    def __init__(self, id: str, x: float, y: float, z: float):
        """Create a new node.

        Parameters
        ----------
        id : str
            Unique ID of the node.
        x : float
            Initial X coordinate of the node.
        y : float
            Initial Y coordinate of the node.
        z : float
            Initial Z coordinate of the node.
        """
        self.id = id
        self._dof_x = Dof(id=(id, 'u'), value=x)
        self._dof_y = Dof(id=(id, 'v'), value=y)
        self._dof_z = Dof(id=(id, 'w'), value=z)

    def dof(self, dof_type: str) -> Dof:
        if dof_type == 'u':
            return self._dof_x
        if dof_type == 'v':
            return self._dof_y
        if dof_type == 'w':
            return self._dof_z
        raise AttributeError('Node has no dof of type \'{}\''.format(dof_type))

    @property
    def ref_x(self) -> float:
        """Gets or sets the x coordinate of the node in the undeformed reference configuration."""
        return self._dof_x.ref_value

    @ref_x.setter
    def ref_x(self, value: float):
        self._dof_x.ref_value = value

    @property
    def ref_y(self) -> float:
        """Gets or sets the y coordinate of the node in the undeformed reference configuration."""
        return self._dof_y.ref_value

    @ref_y.setter
    def ref_y(self, value: float):
        self._dof_y.ref_value = value

    @property
    def ref_z(self) -> float:
        """Gets or sets the z coordinate of the node in the undeformed reference configuration."""
        return self._dof_z.ref_value

    @ref_z.setter
    def ref_z(self, value: float):
        self._dof_z.ref_value = value

    @property
    def x(self) -> float:
        return self._dof_x.value

    @x.setter
    def x(self, value: float):
        self._dof_x.value = value

    @property
    def y(self) -> float:
        return self._dof_y.value

    @y.setter
    def y(self, value: float):
        self._dof_y.value = value

    @property
    def z(self) -> float:
        return self._dof_z.value

    @z.setter
    def z(self, value: float):
        self._dof_z.value = value

    @property
    def u(self) -> float:
        return self._dof_x.delta

    @u.setter
    def u(self, value: float):
        self._dof_x.delta = value

    @property
    def v(self) -> float:
        return self._dof_y.delta

    @v.setter
    def v(self, value: float):
        self._dof_y.delta = value

    @property
    def w(self) -> float:
        return self._dof_z.delta

    @w.setter
    def w(self, value: float):
        self._dof_z.delta = value

    @property
    def fx(self) -> float:
        return self._dof_x.external_force

    @fx.setter
    def fx(self, value: float):
        self._dof_x.external_force = value

    @property
    def fy(self) -> float:
        return self._dof_y.external_force

    @fy.setter
    def fy(self, value: float):
        self._dof_y.external_force = value

    @property
    def fz(self) -> float:
        return self._dof_z.external_force

    @fz.setter
    def fz(self, value: float):
        self._dof_z.external_force = value

    @property
    def external_force(self):
        return np.array([self.fx, self.fy, self.fz])

    @external_force.setter
    def external_force(self, value):
        [self.fx, self.fy, self.fz] = value

    @property
    def ref_location(self) -> List[float]:
        """Gets or sets the z coordinate of the node in the undeformed reference configuration."""
        return np.array([self.ref_x, self.ref_y, self.ref_z])

    @ref_location.setter
    def ref_location(self, value):
        self.ref_x, self.ref_y, self.ref_z = value

    @property
    def location(self):
        return np.array([self.x, self.y, self.z])

    @location.setter
    def location(self, value):
        self.x, self.y, self.z = value

    @property
    def displacement(self):
        return np.array([self.u, self.v, self.w])

    @displacement.setter
    def displacement(self, value):
        self.u, self.v, self.w = value

    @property
    def support(self) -> str:
        result = ''
        if self.support_x:
            result += 'x'
        if self.support_y:
            result += 'y'
        if self.support_z:
            result += 'z'
        return result

    @support.setter
    def support(self, value: str):
        self.support_x = 'x' in value
        self.support_y = 'y' in value
        self.support_z = 'z' in value

    @property
    def support_x(self) -> bool:
        return not self._dof_x.is_active

    @support_x.setter
    def support_x(self, value: bool):
        self._dof_x.is_active = not value

    @property
    def support_y(self) -> bool:
        return not self._dof_y.is_active

    @support_y.setter
    def support_y(self, value: bool):
        self._dof_y.is_active = not value

    @property
    def support_z(self) -> bool:
        return not self._dof_z.is_active

    @support_z.setter
    def support_z(self, value: bool):
        self._dof_z.is_active = not value

    def draw(self, item):
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

        item.add_result('Location undeformed', self.ref_location.tolist())
        item.add_result('Location', self.location.tolist())
        item.add_result('Displacement', (self.location - self.ref_location).tolist())
        item.add_result('Displacement X', float(self.location[0] - self.ref_location[0]))
        item.add_result('Displacement Y', float(self.location[1] - self.ref_location[1]))
        item.add_result('Displacement Z', float(self.location[2] - self.ref_location[2]))

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
