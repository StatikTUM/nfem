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

    def GetActualLocation(self):
        """FIXME"""

        return np.array([self.x, self.y, self.z])

    def GetReferenceLocation(self):
        """FIXME"""

        return np.array([self.reference_x, self.reference_y, self.reference_z])

    def GetDisplacement(self):
        """FIXME"""

        return self.GetReferenceLocation() - self.GetActualLocation()

    def Update(self, dof_type, value):
        """FIXME"""

        if dof_type == 'u':
            self.x = self.reference_x + value
        elif dof_type == 'v':
            self.y = self.reference_y + value
        elif dof_type == 'w':
            self.z = self.reference_z + value
        else:
            raise RuntimeError('Node has no Dof of type {}'.format(dof_type))
