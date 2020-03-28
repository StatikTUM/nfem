"""This module only contains the single load element.

Author: Thomas Oberbichler
"""

import numpy as np

from .element_base import ElementBase

class SingleLoad(ElementBase):
    """FIXME"""

    def __init__(self, id, node, fu, fv, fw):
        """FIXME"""
        self.id = id
        self.node = node
        self.fu = fu
        self.fv = fv
        self.fw = fw

    @property
    def dofs(self):
        """FIXME"""

        node = self.node
        return [node._dof_x, node._dof_y, node._dof_z]

    def calculate_external_forces(self):
        """FIXME"""
        
        return np.array([self.fu, self.fv, self.fw])
