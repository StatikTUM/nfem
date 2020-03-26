"""This module only contains the Node class.

Author: Thomas Oberbichler
"""

import numpy as np
from nfem.dof import Dof


class Node(object):
    """Three dimensional Node providing Dofs for displacements.

    Attributes
    ----------
    id : int or str
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
        id : int or str
            Unique ID of the node.
        x : float
            Initial X coordinate of the node.
        y : float
            Initial Y coordinate of the node.
        z : float
            Initial Z coordinate of the node.
        """
        self.id = id
        self._dof_x = Dof(x)
        self._dof_y = Dof(y)
        self._dof_z = Dof(z)

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
        return np.array([self._dof_x.reference_value, self._dof_y.reference_value, self._dof_z.reference_value])

    @reference_location.setter
    def reference_location(self, value):
        self._dof_x.reference_value, self._dof_y.reference_value, self._dof_z.reference_value = value

    @property
    def location(self):
        return np.array([self._dof_x.value, self._dof_y.value, self._dof_z.value])

    @location.setter
    def location(self, value):
        self._dof_x.value, self._dof_y.value, self._dof_z.value = value

    @property
    def displacement(self):
        return np.array([self._dof_x.delta, self._dof_y.delta, self._dof_z.delta])

    @displacement.setter
    def displacement(self, value):
        self._dof_x.delta, self._dof_y.delta, self._dof_z.delta = value

    def get_dof_state(self, dof_type):
        """Get the current value of the given dof type.

        Parameters
        ----------
        dof_type : string
            Type of the dof.

        Returns
        -------
        value : float
            The current value of the dof type

        Raises
        ------
        AttributeError
            If `dof_type` does not exist.
        """
        return self.dof(dof_type).delta

    def set_dof_state(self, dof_type, value):
        """Update the node according to the value of the given dof type.

        Parameters
        ----------
        dof_type : string
            Type of the Dof.
        value : float
            The value of the given dof.

        Raises
        ------
        AttributeError
            If `dof_type` does not exist.
        """
        self.dof(dof_type).delta = value

    def get_reference_location(self):
        """Location of the node in the reference configuration.

        Returns
        -------
        location : ndarray
            Numpy array containing the reference coordinates X, Y and Z.
        """
        import warnings
        warnings.warn('', warnings.DeprecationWarning)
        return self.reference_location

    def get_actual_location(self):
        """Location of the node in the actual configuration.

        Returns
        -------
        location : ndarray
            Numpy array containing the actual coordinates X, Y and Z.
        """
        import warnings
        warnings.warn('', warnings.DeprecationWarning)
        return self.location

    def get_displacement(self):
        """Displacement of the node in the actual configuration.

        Returns
        -------
        displacement : ndarray
            A numpy array containing the displacements u, v and w.
        """
        import warnings
        warnings.warn('', warnings.DeprecationWarning)
        return self.displacement
