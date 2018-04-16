"""FIXME"""

import numpy as np

class Node(object):
    """FIXME"""

    def __init__(self, id, x, y, z):
        """FIXME"""

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

    def GetActualLocation(self):
        """FIXME"""

        return np.array([self.x, self.y, self.z])

    def GetReferenceLocation(self):
        """FIXME"""

        return np.array([self.reference_x, self.reference_y, self.reference_z])

    def GetDisplacement(self):
        """FIXME"""

        return self.GetReferenceLocation() - self.GetActualLocation()

    def SetDofValue(self, dof_type, value):
        """FIXME"""

        if dof_type == 'u':
            self.u = value
        elif dof_type == 'v':
            self.v =  value
        elif dof_type == 'w':
            self.w = value
        else:
            raise RuntimeError('Node has no Dof of type {}'.format(dof_type))

    def GetDofValue(self, dof_type):
        """Get the current value of the requested dof at this node.

        Parameters
        ----------
        dof_type : str
            type of the dof: possible types: u,v,w

        Returns
        -------
        float
            dof value
        """

        if dof_type == 'u':
            return self.u
        elif dof_type == 'v':
            return self.v
        elif dof_type == 'w':
            return self.w
        else:
            raise RuntimeError('Node has no Dof of type {}'.format(dof_type))

