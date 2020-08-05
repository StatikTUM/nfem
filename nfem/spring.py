"""This module only contains the spring element.

Authors: Thomas Oberbichler
"""

import numpy as np


class Spring:
    """
    Linear spring element.

    Attributes
    ----------
    id : str
        Unique id of the spring element.
    node : Node
        Node.
    kx : float
        Stiffness in x direction.
    ky : float
        Stiffness in y direction.
    kz : float
        Stiffness in z direction.
    dofs

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    gamma(n=1.0)
        Change the photo's gamma exposure.

    """

    def __init__(self, id, node, kx=0, ky=0, kz=0):
        self.id = id
        self.node = node
        self.kx = kx
        self.ky = ky
        self.kz = kz

    @property
    def dofs(self):
        node = self.node
        return [node._dof_x, node._dof_y, node._dof_z]

    def calculate_elastic_stiffness_matrix(self):
        return np.array([[self.kx, 0, 0], [0, self.ky, 0], [0, 0, self.kz]])

    def calculate_material_stiffness_matrix(self):
        return None

    def calculate_initial_displacement_stiffness_matrix(self):
        return None

    def calculate_geometric_stiffness_matrix(self, linear=False):
        return None

    def calculate_stiffness_matrix(self):
        return self.calculate_elastic_stiffness_matrix()

    def calculate_internal_forces(self):
        return np.array([
            self.kx * self.node.u,
            self.ky * self.node.v,
            self.kz * self.node.w,
        ])

    def draw(self, canvas):
        location = self.node.location
        if self.kx != 0:
            canvas.spring(location, 'x')
        if self.ky != 0:
            canvas.spring(location, 'y')
        if self.kz != 0:
            canvas.spring(location, 'z')
