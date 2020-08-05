"""This module only contains the Node class.

Author: Thomas Oberbichler
"""

import numpy as np
from nfem.dof import Dof


class Node:
    """Three dimensional Node providing Dofs for displacements.

    Attributes
    ----------
    id : str
        Unique ID.
    reference_x : float
        Reference X coordinate.
    reference_y : float
        Reference Y coordinate.
    reference_z : float
        Reference Z coordinate.
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

    def __init__(self, id, x, y, z):
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

    def dof(self, dof_type):
        if dof_type == 'u':
            return self._dof_x
        if dof_type == 'v':
            return self._dof_y
        if dof_type == 'w':
            return self._dof_z
        raise AttributeError('Node has no dof of type \'{}\''.format(dof_type))

    @property
    def reference_x(self):
        return self._dof_x.reference_value

    @reference_x.setter
    def reference_x(self, value):
        self._dof_x.reference_value = value

    @property
    def reference_y(self):
        return self._dof_y.reference_value

    @reference_y.setter
    def reference_y(self, value):
        self._dof_y.reference_value = value

    @property
    def reference_z(self):
        return self._dof_z.reference_value

    @reference_z.setter
    def reference_z(self, value):
        self._dof_z.reference_value = value

    @property
    def x(self):
        return self._dof_x.value

    @x.setter
    def x(self, value):
        self._dof_x.value = value

    @property
    def y(self):
        return self._dof_y.value

    @y.setter
    def y(self, value):
        self._dof_y.value = value

    @property
    def z(self):
        return self._dof_z.value

    @z.setter
    def z(self, value):
        self._dof_z.value = value

    @property
    def u(self):
        return self._dof_x.delta

    @u.setter
    def u(self, value):
        self._dof_x.delta = value

    @property
    def v(self):
        return self._dof_y.delta

    @v.setter
    def v(self, value):
        self._dof_y.delta = value

    @property
    def w(self):
        return self._dof_z.delta

    @w.setter
    def w(self, value):
        self._dof_z.delta = value

    @property
    def fx(self):
        return self._dof_x.external_force

    @fx.setter
    def fx(self, value):
        self._dof_x.external_force = value

    @property
    def fy(self):
        return self._dof_y.external_force

    @fy.setter
    def fy(self, value):
        self._dof_y.external_force = value

    @property
    def fz(self):
        return self._dof_z.external_force

    @fz.setter
    def fz(self, value):
        self._dof_z.external_force = value

    @property
    def external_force(self):
        return np.array([self.fx, self.fy, self.fz])

    @external_force.setter
    def external_force(self, value):
        [self.fx, self.fy, self.fz] = value

    @property
    def reference_location(self):
        return np.array([self.reference_x, self.reference_y, self.reference_z])

    @reference_location.setter
    def reference_location(self, value):
        self.reference_x, self.reference_y, self.reference_z = value

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
    def support(self):
        result = ''
        if self.support_x:
            result += 'x'
        if self.support_y:
            result += 'y'
        if self.support_z:
            result += 'z'
        return result

    @support.setter
    def support(self, value):
        self.support_x = 'x' in value
        self.support_y = 'y' in value
        self.support_z = 'z' in value

    @property
    def support_x(self):
        return not self._dof_x.is_active

    @support_x.setter
    def support_x(self, value):
        self._dof_x.is_active = not value

    @property
    def support_y(self):
        return not self._dof_y.is_active

    @support_y.setter
    def support_y(self, value):
        self._dof_y.is_active = not value

    @property
    def support_z(self):
        return not self._dof_z.is_active

    @support_z.setter
    def support_z(self, value):
        self._dof_z.is_active = not value

    def draw(self, canvas):
        canvas.support(
            location=self.reference_location,
            direction=self.support,
            layer=10,
            color='gray',
        )
        canvas.support(
            location=self.location,
            direction=self.support,
            layer=20,
            color='red',
        )
        canvas.point(
            location=self.reference_location,
            layer=10,
            color='gray',
        )
        canvas.text(
            location=self.location,
            text=self.id,
            layer=21,
            color='gray',
        )
        canvas.point(
            location=self.location,
            layer=20,
            color='red',
        )

        if np.linalg.norm(self.external_force) > 1e-8:
            direction = self.external_force
            direction /= np.linalg.norm(direction)

            canvas.arrow(
                location=(self.location - direction),
                direction=direction,
                layer=23,
                color='blue',
            )
