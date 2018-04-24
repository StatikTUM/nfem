"""This module only contains the Node class.

Author: Thomas Oberbichler
"""

import numpy as np

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
        self.x = x
        self.y = y
        self.z = z
        self.reference_x = x
        self.reference_y = y
        self.reference_z = z

    @property
    def u(self):
        return self.x - self.reference_x

    @u.setter
    def u(self, value):
        self.x = self.reference_x + value

    @property
    def v(self):
        return self.y - self.reference_y

    @v.setter
    def v(self, value):
        self.y = self.reference_y + value
    
    @property
    def w(self):
        return self.z - self.reference_z

    @w.setter
    def w(self, value):
        self.z = self.reference_z + value

    def get_reference_location(self):
        """Location of the node in the reference configuration.

        Returns
        -------
        location : ndarray
            Numpy array containing the reference coordinates X, Y and Z.
        """
        x = self.reference_x
        y = self.reference_y
        z = self.reference_z

        return np.array([x, y, z], dtype=float)

    def get_actual_location(self):
        """Location of the node in the actual configuration.

        Returns
        -------
        location : ndarray
            Numpy array containing the actual coordinates X, Y and Z.
        """
        x = self.x
        y = self.y
        z = self.z

        return np.array([x, y, z], dtype=float)

    def get_displacement(self):
        """Displacement of the node in the actual configuration.

        Returns
        -------
        displacement : ndarray
            A numpy array containing the displacements u, v and w.
        """
        return self.get_reference_location() - self.get_actual_location()

    def update(self, dof_type, value):
        """
        .. note:: Deprecated
                  Use `SetDofState` instead
        """
        raise DeprecationWarning('Use `SetDofState` instead')
        if dof_type == 'u':
            self.u = value
        elif dof_type == 'v':
            self.v =  value
        elif dof_type == 'w':
            self.w = value
        else:
            raise RuntimeError('Node has no Dof of type {}'.format(dof_type))

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
        if dof_type == 'u':
            return self.u
        if dof_type == 'v':
            return self.v
        if dof_type == 'w':
            return self.w

        raise AttributeError('Node has no dof of type \'{}\''.format(dof_type))

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
        if dof_type == 'u':
            self.u = value
        elif dof_type == 'v':
            self.v = value
        elif dof_type == 'w':
            self.w = value
        else:
            raise AttributeError('Node has no dof of type \'{}\''.format(dof_type))
