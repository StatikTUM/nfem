"""This module only contains the Element base class.

Author: Thomas Oberbichler
"""

class ElementBase(object):
    """FIXME"""

    @property
    def dofs(self):
        """FIXME"""
        raise NotImplementedError

    def calculate_elastic_stiffness_matrix(self):
        """FIXME"""
        return None

    def calculate_geometric_stiffness_matrix(self):
        """FIXME"""
        return None

    def calculate_stiffness_matrix(self):
        """FIXME"""
        return None

    def calculate_external_forces(self):
        """FIXME"""
        return None

    def calculate_internal_forces(self):
        """FIXME"""
        return None
