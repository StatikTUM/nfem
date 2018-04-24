"""This module only contains the truss element.

Authors: Thomas Oberbichler, Klaus Sautter
"""

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

    @property
    def dofs(self):
        """FIXME"""

        a_id = self.node_a.id
        b_id = self.node_b.id

        return [(a_id, 'u'), (a_id, 'v'), (a_id, 'w'), (b_id, 'u'), (b_id, 'v'), (b_id, 'w')]

    def get_reference_vector(self):
        """FIXME"""

        reference_a = self.node_a.get_reference_location()
        reference_b = self.node_b.get_reference_location()

        return reference_b - reference_a

    def get_actual_vector(self):
        """FIXME"""

        actual_a = self.node_a.get_actual_location()
        actual_b = self.node_b.get_actual_location()

        return actual_b - actual_a

    def get_reference_length(self):
        """FIXME"""

        reference_a = self.node_a.get_reference_location()
        reference_b = self.node_b.get_reference_location()

        return la.norm(reference_b - reference_a)

    def get_actual_length(self):
        """FIXME"""

        actual_a = self.node_a.get_actual_location()
        actual_b = self.node_b.get_actual_location()
        
        return la.norm(actual_b - actual_a)

    def calculate_elastic_stiffness_matrix(self):
        """FIXME"""

        reference_a = self.node_a.get_reference_location()
        reference_b = self.node_b.get_reference_location()

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

    def calculate_geometric_stiffness_matrix(self):
        """FIXME"""

        E = self.youngs_modulus
        A = self.area

        prestress = self.prestress

        reference_length = self.get_reference_length()

        dx, dy, dz = self.get_reference_vector()
        du, dv, dw = self.get_actual_vector() - self.get_reference_vector()

        e_gl = self.calculate_green_lagrange_strain()

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

    def calculate_stiffness_matrix(self):
        """FIXME"""

        element_k_e = self.calculate_elastic_stiffness_matrix()
        element_k_g = self.calculate_geometric_stiffness_matrix()

        return element_k_e + element_k_g

    def calculate_transformation_matrix(self):
        """FIXME"""

        direction = self.get_actual_vector()
        direction /= la.norm(direction)

        transformation_matrix = np.zeros((6,6))
        transformation_matrix[:3, 0] = direction
        transformation_matrix[3:, 3] = direction
        
        return transformation_matrix

    def calculate_green_lagrange_strain(self):
        """FIXME"""

        reference_length = self.get_reference_length()
        actual_length = self.get_actual_length()

        e_gl = (actual_length**2 - reference_length**2) / (2 * reference_length**2)

        return e_gl

    def calculate_internal_forces(self):
        """FIXME"""

        transformation_matrix = self.calculate_transformation_matrix()

        e_gl = self.calculate_green_lagrange_strain()

        E = self.youngs_modulus
        A = self.area
        prestress = self.prestress

        reference_length = self.get_reference_length()
        actual_length = self.get_actual_length()

        deformation_gradient = actual_length / reference_length

        normal_force = (E * e_gl + prestress) * A * deformation_gradient 

        local_internal_forces = np.zeros(6)
        local_internal_forces[0] = -normal_force
        local_internal_forces[3] = normal_force

        global_internal_forces = transformation_matrix @ local_internal_forces

        return global_internal_forces

