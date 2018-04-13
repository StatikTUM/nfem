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

    def CalculateElasticStiffnessMatrix(self):
        """FIXME"""

        reference_a = self.node_a.GetReferenceLocation()
        reference_b = self.node_b.GetReferenceLocation()

        reference_length = la.norm(reference_b - reference_a)

        (dx, dy, dz) = reference_b - reference_a

        EA = self.youngs_modulus * self.area

        L3 = reference_length**3

        k_e = np.empty((6, 6))

        k_e[0, 0] = (EA * dx * dx) / L3
        k_e[0, 1] = (EA * dx * dy) / L3
        k_e[0, 2] = (EA * dx * dz) / L3
        k_e[0, 3] = -k_e[0, 0]
        k_e[0, 4] = -k_e[0, 1]
        k_e[0, 5] = -k_e[0, 2]
        k_e[1, 1] = (EA * dy * dy) / L3
        k_e[1, 2] = (EA * dy * dz) / L3
        k_e[1, 3] = k_e[0, 4]
        k_e[1, 4] = -k_e[1, 1]
        k_e[1, 5] = -k_e[1, 2]
        k_e[2, 2] = (EA * dz * dz) / L3
        k_e[2, 3] = -k_e[0, 2]
        k_e[2, 4] = -k_e[1, 2]
        k_e[2, 5] = -k_e[2, 2]
        k_e[3, 3] = k_e[0, 0]
        k_e[3, 4] = k_e[0, 1]
        k_e[3, 5] = k_e[0, 2]
        k_e[4, 4] = k_e[1, 1]
        k_e[4, 5] = k_e[1, 2]
        k_e[5, 5] = k_e[2, 2]

        # symmetry

        k_e[1, 0] = k_e[0, 1]
        k_e[2, 0] = k_e[0, 2]
        k_e[2, 1] = k_e[1, 2]
        k_e[3, 0] = k_e[0, 3]
        k_e[3, 1] = k_e[1, 3]
        k_e[3, 2] = k_e[2, 3]
        k_e[4, 0] = k_e[0, 4]
        k_e[4, 1] = k_e[1, 4]
        k_e[4, 2] = k_e[2, 4]
        k_e[4, 3] = k_e[3, 4]
        k_e[5, 0] = k_e[0, 5]
        k_e[5, 1] = k_e[1, 5]
        k_e[5, 2] = k_e[2, 5]
        k_e[5, 3] = k_e[3, 5]
        k_e[5, 4] = k_e[4, 5]

        return k_e

    def CalculateGeometricStiffnessMatrix(self):
        """FIXME"""

        E = self.youngs_modulus
        A = self.area

        prestress = self.prestress

        reference_a = self.node_a.GetReferenceLocation()
        reference_b = self.node_b.GetReferenceLocation()
        reference_ab = reference_b - reference_a
        reference_length = la.norm(reference_ab)

        actual_a = self.node_a.GetActualLocation()
        actual_b = self.node_b.GetActualLocation()
        actual_ab = actual_b - actual_a
        actual_length = la.norm(actual_ab)

        dx, dy, dz = reference_ab
        du, dv, dw = actual_ab - reference_ab

        e_gl = (actual_length**2 - reference_length**2) / (2.00 * reference_length**2)

        K_sigma = ((E * A * e_gl) / reference_length) + ((prestress * A) / reference_length)
        K_uij = (E * A) / reference_length**3

        k_g = np.empty((6, 6))

        k_g[0, 0] = K_sigma + K_uij * (2 * du * dx + du * du)
        k_g[0, 1] = K_uij * (dx * dv + dy * du + du * dv)
        k_g[0, 2] = K_uij * (dx * dw + dz * du + du * dw)
        k_g[0, 3] = -k_g[0, 0]
        k_g[0, 4] = -k_g[0, 1]
        k_g[0, 5] = -k_g[0, 2]
        k_g[1, 1] = K_sigma + K_uij * (2 * dv * dy + dv * dv)
        k_g[1, 2] = K_uij * (dy * dw + dz * dv + dv * dw)
        k_g[1, 3] = k_g[0, 4]
        k_g[1, 4] = -k_g[1, 1]
        k_g[1, 5] = -k_g[1, 2]
        k_g[2, 2] = K_sigma + K_uij * (2 * dw * dz + dw * dw)
        k_g[2, 3] = -k_g[0, 2]
        k_g[2, 4] = -k_g[1, 2]
        k_g[2, 5] = -k_g[2, 2]
        k_g[3, 3] = k_g[0, 0]
        k_g[3, 4] = k_g[0, 1]
        k_g[3, 5] = k_g[0, 2]
        k_g[4, 4] = k_g[1, 1]
        k_g[4, 5] = k_g[1, 2]
        k_g[5, 5] = k_g[2, 2]

        # symmetry

        k_g[1, 0] = k_g[0, 1]
        k_g[2, 0] = k_g[0, 2]
        k_g[2, 1] = k_g[1, 2]
        k_g[3, 0] = k_g[0, 3]
        k_g[3, 1] = k_g[1, 3]
        k_g[3, 2] = k_g[2, 3]
        k_g[4, 0] = k_g[0, 4]
        k_g[4, 1] = k_g[1, 4]
        k_g[4, 2] = k_g[2, 4]
        k_g[4, 3] = k_g[3, 4]
        k_g[5, 0] = k_g[0, 5]
        k_g[5, 1] = k_g[1, 5]
        k_g[5, 2] = k_g[2, 5]
        k_g[5, 3] = k_g[3, 5]
        k_g[5, 4] = k_g[4, 5]

        return k_g

    def CalculateStiffnessMatrix(self):
        """FIXME"""

        element_k_e = self.CalculateElasticStiffnessMatrix()
        element_k_g = self.CalculateGeometricStiffnessMatrix()

        return element_k_e + element_k_g

    def CreateTransformationMatrix(self):
        
        location_a = self.node_a.GetActualLocation()
        location_b = self.node_b.GetActualLocation()

        direction_longitudinal = location_b-location_a
        norm_direction_longitudinal = la.norm(direction_longitudinal)
        direction_longitudinal = direction_longitudinal/norm_direction_longitudinal

        transformation_matrix = np.zeros((6,6))
        transformation_matrix[0:3,0] = direction_longitudinal
        transformation_matrix[3:6,3] = direction_longitudinal
        
        return transformation_matrix

    def CalculateGreenLagrangeStrain(self):
        reference_a = self.node_a.GetReferenceLocation()
        reference_b = self.node_b.GetReferenceLocation()
        reference_ab = reference_b - reference_a

        actual_a = self.node_a.GetActualLocation()
        actual_b = self.node_b.GetActualLocation()
        actual_ab = actual_b - actual_a

        dx, dy, dz = reference_ab
        du, dv, dw = actual_ab - reference_ab

        L = la.norm([dx, dy, dz])
        l = la.norm([dx + du, dy + dv, dz + dw])

        e_gl = (l**2 - L**2) / (2.00 * L**2)
        return e_gl

    def CalculateInternalForces(self):
        transformation_matrix = self.CreateTransformationMatrix()
        e_gl = self.CalculateGreenLagrangeStrain()

        E = self.youngs_modulus
        A = self.area
        prestress = self.prestress

        dx, dy, dz = reference_ab
        du, dv, dw = actual_ab - reference_ab

        L = la.norm([dx, dy, dz])
        l = la.norm([dx + du, dy + dv, dz + dw])

        normal_force = (E * e_gl + prestress) * A * deformation_gradient 

        local_internal_forces = np.zeros(6)
        local_internal_forces[0] = -normal_force
        local_internal_forces[3] = normal_force

        global_internal_forces = transformation_matrix @ local_internal_forces
        return global_internal_forces

