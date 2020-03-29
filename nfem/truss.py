"""This module only contains the truss element.

Authors: Thomas Oberbichler, Klaus Sautter
"""

import numpy as np
import numpy.linalg as la


class Truss:
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
        node_a = self.node_a
        node_b = self.node_b

        return [node_a._dof_x, node_a._dof_y, node_a._dof_z, node_b._dof_x, node_b._dof_y, node_b._dof_z]

    def get_reference_vector(self):
        """FIXME"""

        reference_a = self.node_a.reference_location
        reference_b = self.node_b.reference_location

        return reference_b - reference_a

    def get_actual_vector(self):
        """FIXME"""

        actual_a = self.node_a.location
        actual_b = self.node_b.location

        return actual_b - actual_a

    def get_reference_length(self):
        """FIXME"""

        reference_a = self.node_a.reference_location
        reference_b = self.node_b.reference_location

        return la.norm(reference_b - reference_a)

    def get_actual_length(self):
        """FIXME"""

        actual_a = self.node_a.location
        actual_b = self.node_b.location

        return la.norm(actual_b - actual_a)

    def get_reference_transformMatrix(self):
        """ Transformation matrix for the reference configuration.

        Returns
        -------
        reference_transform : ndarray
            Transformation matrix.
        """
        direction = self.get_reference_vector()
        direction = direction / la.norm(direction)

        reference_transform = np.zeros((2, 6))
        reference_transform[0, :3] = direction
        reference_transform[1, 3:] = direction

        return reference_transform

    def get_actual_transform_matrix(self):
        """ Transformation matrix for the actual configuration.

        Returns
        -------
        actual_transform : ndarray
            Transformation matrix.
        """
        direction = self.get_actual_vector()
        direction = direction / la.norm(direction)

        actual_transform = np.zeros((2, 6))
        actual_transform[0, :3] = direction
        actual_transform[1, 3:] = direction

        return actual_transform

    def calculate_elastic_stiffness_matrix(self):
        """FIXME"""

        e = self.youngs_modulus
        a = self.area
        reference_length = self.get_reference_length()
        reference_transform = self.get_reference_transformMatrix()

        k_e = e * a / reference_length

        return reference_transform.T @ [[k_e, -k_e], [-k_e, k_e]] @ reference_transform

    def calculate_material_stiffness_matrix(self):
        """FIXME"""

        e = self.youngs_modulus
        a = self.area
        actual_length = self.get_actual_length()
        reference_length = self.get_reference_length()
        actual_transform = self.get_actual_transform_matrix()

        k_m = e * a / reference_length * (actual_length / reference_length)**2

        return actual_transform.T @ [[k_m, -k_m], [-k_m, k_m]] @ actual_transform

    def calculate_initial_displacement_stiffness_matrix(self):
        """FIXME"""

        k_m = self.calculate_material_stiffness_matrix()
        k_e = self.calculate_elastic_stiffness_matrix()

        return k_m - k_e

    def calculate_geometric_stiffness_matrix(self, linear=False):
        e = self.youngs_modulus
        a = self.area
        prestress = self.prestress
        reference_length = self.get_reference_length()

        if linear:
            # FIXME this is not a clean solution to solve the LPB issue
            epsilon = self.calculate_linear_strain()
        else:
            epsilon = self.calculate_green_lagrange_strain()

        k_g = e * a / reference_length * epsilon + prestress * a / reference_length

        return np.array([[k_g, 0, 0, -k_g, 0, 0],
                         [0,  k_g, 0, 0, -k_g, 0],
                         [0, 0,  k_g, 0, 0, -k_g],
                         [-k_g, 0, 0,  k_g, 0, 0],
                         [0, -k_g, 0, 0,  k_g, 0],
                         [0, 0, -k_g, 0, 0, k_g]])

    def calculate_stiffness_matrix(self):
        """FIXME"""

        element_k_m = self.calculate_material_stiffness_matrix()
        element_k_g = self.calculate_geometric_stiffness_matrix()

        return element_k_m + element_k_g

    def calculate_green_lagrange_strain(self):
        """FIXME"""

        reference_length = self.get_reference_length()
        actual_length = self.get_actual_length()

        e_gl = (actual_length**2 - reference_length**2) / (2 * reference_length**2)

        return e_gl

    def calculate_linear_strain(self):
        """FIXME"""

        reference_vec = self.get_reference_vector()
        actual_vec = self.get_actual_vector()

        reference_length = self.get_reference_length()

        # project actual on reference
        projected_l = reference_vec @ actual_vec / reference_length

        e_lin = (projected_l - reference_length) / reference_length

        return e_lin

    def calculate_internal_forces(self):
        """FIXME"""

        e_gl = self.calculate_green_lagrange_strain()

        E = self.youngs_modulus
        A = self.area
        prestress = self.prestress

        reference_length = self.get_reference_length()
        actual_length = self.get_actual_length()

        deformation_gradient = actual_length / reference_length

        normal_force = (E * e_gl + prestress) * A * deformation_gradient

        local_internal_forces = [-normal_force, normal_force]

        actual_transform = self.get_actual_transform_matrix()

        global_internal_forces = actual_transform.T @ local_internal_forces

        return global_internal_forces
