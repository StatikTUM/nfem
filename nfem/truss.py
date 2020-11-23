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

    def get_ref_vector(self):
        """FIXME"""

        ref_a = self.node_a.ref_location
        ref_b = self.node_b.ref_location

        return ref_b - ref_a

    def get_act_vector(self):
        """FIXME"""

        act_a = self.node_a.location
        act_b = self.node_b.location

        return act_b - act_a

    def get_ref_length(self):
        """FIXME"""

        ref_a = self.node_a.ref_location
        ref_b = self.node_b.ref_location

        return la.norm(ref_b - ref_a)

    def get_act_length(self):
        """FIXME"""

        act_a = self.node_a.location
        act_b = self.node_b.location

        return la.norm(act_b - act_a)

    def get_ref_transformMatrix(self):
        """ Transformation matrix for the reference configuration.

        Returns
        -------
        ref_transform : ndarray
            Transformation matrix.
        """
        direction = self.get_ref_vector()
        direction = direction / la.norm(direction)

        ref_transform = np.zeros((2, 6))
        ref_transform[0, :3] = direction
        ref_transform[1, 3:] = direction

        return ref_transform

    def get_act_transform_matrix(self):
        """ Transformation matrix for the actual configuration.

        Returns
        -------
        act_transform : ndarray
            Transformation matrix.
        """
        direction = self.get_act_vector()
        direction = direction / la.norm(direction)

        act_transform = np.zeros((2, 6))
        act_transform[0, :3] = direction
        act_transform[1, 3:] = direction

        return act_transform

    def calculate_elastic_stiffness_matrix(self):
        """FIXME"""

        e = self.youngs_modulus
        a = self.area
        ref_length = self.get_ref_length()
        ref_transform = self.get_ref_transformMatrix()

        k_e = e * a / ref_length

        return ref_transform.T @ [[k_e, -k_e], [-k_e, k_e]] @ ref_transform

    def calculate_material_stiffness_matrix(self):
        """FIXME"""

        e = self.youngs_modulus
        a = self.area
        act_length = self.get_act_length()
        ref_length = self.get_ref_length()
        act_transform = self.get_act_transform_matrix()

        k_m = e * a / ref_length * (act_length / ref_length)**2

        return act_transform.T @ [[k_m, -k_m], [-k_m, k_m]] @ act_transform

    def calculate_initial_displacement_stiffness_matrix(self):
        """FIXME"""

        k_m = self.calculate_material_stiffness_matrix()
        k_e = self.calculate_elastic_stiffness_matrix()

        return k_m - k_e

    def calculate_geometric_stiffness_matrix(self, linear=False):
        e = self.youngs_modulus
        a = self.area
        prestress = self.prestress
        ref_length = self.get_ref_length()

        if linear:
            # FIXME this is not a clean solution to solve the LPB issue
            epsilon = self.calculate_linear_strain()
        else:
            epsilon = self.calculate_green_lagrange_strain()

        k_g = e * a / ref_length * epsilon + prestress * a / ref_length

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

        ref_length = self.get_ref_length()
        act_length = self.get_act_length()

        e_gl = (act_length**2 - ref_length**2) / (2 * ref_length**2)

        return e_gl

    def calculate_linear_strain(self):
        """FIXME"""

        ref_vec = self.get_ref_vector()
        act_vec = self.get_act_vector()

        ref_length = self.get_ref_length()

        # project actual on reference
        projected_l = ref_vec @ act_vec / ref_length

        e_lin = (projected_l - ref_length) / ref_length

        return e_lin

    def calculate_internal_forces(self):
        """FIXME"""

        e_gl = self.calculate_green_lagrange_strain()

        E = self.youngs_modulus
        A = self.area
        prestress = self.prestress

        ref_length = self.get_ref_length()
        act_length = self.get_act_length()

        deformation_gradient = act_length / ref_length

        normal_force = (E * e_gl + prestress) * A * deformation_gradient

        local_internal_forces = [-normal_force, normal_force]

        act_transform = self.get_act_transform_matrix()

        global_internal_forces = act_transform.T @ local_internal_forces

        return global_internal_forces

    def draw(self, canvas):
        canvas.line(
            a=self.node_a.ref_location.tolist(),
            b=self.node_b.ref_location.tolist(),
            layer=10,
            color='gray',
        )
        canvas.line(
            a=self.node_a.location.tolist(),
            b=self.node_b.location.tolist(),
            layer=20,
            color='red',
        )
        canvas.text(
            location=((self.node_a.location + self.node_b.location) / 2).tolist(),
            text=self.id,
            layer=22,
        )
