"""FIXME"""

import numpy as np
import numpy.linalg as la

from .element_base import ElementBase

class Truss(ElementBase):
    """FIXME"""

    def __init__(self, id, node_a, node_b, youngs_modulus, area, prestress=0):
        """FIXME"""

        self.id = id
        self.node_a = node_a
        self.node_b = node_b
        self.youngs_modulus = youngs_modulus
        self.area = area
        self.prestress = prestress

    def Dofs(self):
        """FIXME"""

        a_id = self.node_a.id
        b_id = self.node_b.id

        return [(a_id, 'u'), (a_id, 'v'), (a_id, 'w'), (b_id, 'u'), (b_id, 'v'), (b_id, 'w')]

    def GetReferenceVector(self):
        """FIXME"""

        reference_a = self.node_a.GetReferenceLocation()
        reference_b = self.node_b.GetReferenceLocation()

        return reference_b - reference_a

    def GetActualVector(self):
        """FIXME"""

        actual_a = self.node_a.GetActualLocation()
        actual_b = self.node_b.GetActualLocation()

        return actual_b - actual_a

    def GetReferenceLength(self):
        """FIXME"""

        reference_a = self.node_a.GetReferenceLocation()
        reference_b = self.node_b.GetReferenceLocation()

        return la.norm(reference_b - reference_a)

    def GetActualLength(self):
        """FIXME"""

        actual_a = self.node_a.GetActualLocation()
        actual_b = self.node_b.GetActualLocation()
        
        return la.norm(actual_b - actual_a)

    def GetReferenceTransformMatrix(self):
        """ Transformation matrix for the reference configuration.
        """
        direction = self.GetReferenceVector()
        direction = direction / la.norm(direction)

        reference_transform = np.zeros((2, 6))
        reference_transform[0, :3] = direction
        reference_transform[1, 3:] = direction

        return reference_transform

    def GetActualTransformMatrix(self):
        """ Transformation matrix for the actual configuration.
        """
        direction = self.GetActualVector()
        direction = direction / la.norm(direction)

        actual_transform = np.zeros((2, 6))
        actual_transform[0, :3] = direction
        actual_transform[1, 3:] = direction

        return actual_transform

    def CalculateElasticStiffnessMatrix(self):
        """FIXME"""

        e = self.youngs_modulus
        a = self.area
        reference_length = self.GetReferenceLength()
        reference_transform = self.GetReferenceTransformMatrix()

        k_e = e * a / reference_length

        return reference_transform.T @ [[ k_e, -k_e],
                                        [-k_e,  k_e]] @ reference_transform

    def CalculateMaterialStiffnessMatrix(self):
        """FIXME"""

        e = self.youngs_modulus
        a = self.area
        actual_length = self.GetReferenceLength() 
        reference_length = self.GetReferenceLength()
        actual_transform = self.GetActualTransformMatrix()

        k_m = e * a / reference_length * (actual_length / reference_length)**2

        return actual_transform.T @ [[ k_m, -k_m],
                                     [-k_m,  k_m]] @ actual_transform

    def CalculateInitialDisplacementStiffnessMatrix(self):
        """FIXME"""

        k_m = self.CalculateMaterialStiffnessMatrix()
        k_e = self.CalculateElasticStiffnessMatrix()

        return k_m - k_e

    def CalculateGeometricStiffnessMatrix(self):
        e = self.youngs_modulus
        a = self.area
        prestress = self.prestress
        reference_length = self.GetReferenceLength()

        e_gl = self.CalculateGreenLagrangeStrain()

        k_g = e * a / reference_length * e_gl + prestress * a / reference_length

        return np.array([[ k_g,    0,    0, -k_g,    0,    0],
                         [   0,  k_g,    0,    0, -k_g,    0],
                         [   0,    0,  k_g,    0,    0, -k_g],
                         [-k_g,    0,    0,  k_g,    0,    0],
                         [   0, -k_g,    0,    0,  k_g,    0],
                         [   0,    0, -k_g,    0,    0,  k_g]])

    def CalculateStiffnessMatrix(self):
        """FIXME"""

        element_k_e = self.CalculateElasticStiffnessMatrix()
        element_k_g = self.CalculateGeometricStiffnessMatrix()

        return element_k_e + element_k_g

    def CalculateGreenLagrangeStrain(self):
        """FIXME"""

        reference_length = self.GetReferenceLength()
        actual_length = self.GetActualLength()

        e_gl = (actual_length**2 - reference_length**2) / (2 * reference_length**2)

        return e_gl

    def CalculateInternalForces(self):
        """FIXME"""

        e_gl = self.CalculateGreenLagrangeStrain()

        E = self.youngs_modulus
        A = self.area
        prestress = self.prestress

        reference_length = self.GetReferenceLength()
        actual_length = self.GetActualLength()

        deformation_gradient = actual_length / reference_length

        normal_force = (E * e_gl + prestress) * A * deformation_gradient 

        local_internal_forces = [-normal_force, normal_force]

        actual_transform = self.GetActualTransformMatrix()

        global_internal_forces = actual_transform.T @ local_internal_forces

        return global_internal_forces
