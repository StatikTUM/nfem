"""FIXME"""

class ElementBase(object):
    """FIXME"""

    def Dofs(self):
        """FIXME"""
        raise NotImplementedError

    def CalculateElasticStiffnessMatrix(self):
        """FIXME"""
        return None

    def CalculateGeometricStiffnessMatrix(self):
        """FIXME"""
        return None

    def CalculateStiffnessMatrix(self):
        """FIXME"""
        return None

    def CalculateExternalForces(self):
        """FIXME"""
        return None

    def CalculateInternalForces(self):
        """FIXME"""
        return None
