from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import proj3d

import numpy as np


class Arrow3D(FancyArrowPatch):
    """from https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Circle3D(Circle):
    def __init__(self, x, y, z, r, *args, **kwargs):
        Circle.__init__(self, (0, 0), r, *args, **kwargs)
        self._verts3d = x, y, z

    def draw(self, renderer):
        x3d, y3d, z3d = self._verts3d
        x, y, z = proj3d.proj_transform(x3d, y3d, z3d, renderer.M)
        self.center = x, y
        Circle.draw(self, renderer)


def get_force_arrow(x, y, z, fx, fy, fz, length, *args, **kwargs):
    delta = np.array([fx, fy, fz])
    dnorm = np.linalg.norm(delta)
    if dnorm < 1e-8:
        return None
    delta = delta * length / dnorm
    return Arrow3D([x, x-delta[0]], [y, y-delta[1]], [z, z-delta[2]], *args, mutation_scale=15, lw=1.5, arrowstyle="<|-", **kwargs)


def get_dof_arrow(x, y, z, dx, dy, dz, length, *args, **kwargs):
    delta = np.array([dx, dy, dz])
    delta = delta/np.linalg.norm(delta)*length
    return Arrow3D([x, x+delta[0]], [y, y+delta[1]], [z, z+delta[2]], *args, mutation_scale=15, lw=1.5, arrowstyle="-|>", **kwargs)


def get_sphere(x, y, z, r, *args, **kwargs):
    return Circle3D(x, y, z, r, *args, **kwargs)


def get_tet4_polygons(x, y, z, h, direction):
    d = h/2
    node_1 = (x, y, z)
    if direction == 'u':
        node_2 = (x-h, y-d, z-d)
        node_3 = (x-h, y+d, z-d)
        node_4 = (x-h, y+d, z+d)
        node_5 = (x-h, y-d, z+d)
    elif direction == 'v':
        node_2 = (x-d, y-h, z-d)
        node_3 = (x-d, y-h, z+d)
        node_4 = (x+d, y-h, z+d)
        node_5 = (x+d, y-h, z-d)
    elif direction == 'w':
        node_2 = (x-d, y-d, z-h)
        node_3 = (x-d, y+d, z-h)
        node_4 = (x+d, y+d, z-h)
        node_5 = (x+d, y-d, z-h)

    poly_1 = [node_1, node_2, node_3]
    poly_2 = [node_1, node_3, node_4]
    poly_3 = [node_1, node_3, node_5]
    poly_4 = [node_1, node_5, node_2]
    poly_5 = [node_2, node_3, node_4, node_5]

    return poly_1, poly_2, poly_3, poly_4, poly_5
