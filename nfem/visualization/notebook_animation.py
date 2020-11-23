"""This module contains helpers for visualize data.

Authors: Klaus Sautter, Thomas Oberbichler, Armin Geiser
"""

from IPython.display import HTML
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as anim

from nfem.visualization.plot_symbols import get_force_arrow, get_tet4_polygons, get_dof_arrow

from nfem.spring import Spring
from nfem.truss import Truss
from nfem.model_status import ModelStatus


def get_bounding_box(models):
    nodes = [node for model in models for node in model.nodes]

    min_x = min(node.x for node in nodes)
    max_x = max(node.x for node in nodes)

    min_y = min(node.y for node in nodes)
    max_y = max(node.y for node in nodes)

    min_z = min(node.z for node in nodes)
    max_z = max(node.z for node in nodes)

    return min_x, max_x, min_y, max_y, min_z, max_z


def plot_scaled_model(ax, model, color, **options):
    scaling_factor = options.get('plot/scaling_factor', None)
    if not scaling_factor:
        # autoscaling max u to 10% of bounding_box
        bounding_box = get_bounding_box([model.get_initial_model()])
        min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
        max_delta = max(max_x-min_x, max_y-min_y, max_z-min_z)
        max_u = max(abs(node.u) for node in model.nodes)
        max_v = max(abs(node.v) for node in model.nodes)
        max_w = max(abs(node.w) for node in model.nodes)
        max_def = max(max_u, max_v, max_w)
        scaling_factor = max_delta / max_def * 0.1

    lines = list()

    for element in model.elements:
        if type(element) == Truss:
            node_a = element.node_a
            node_b = element.node_b

            b = [node_b.ref_x+scaling_factor*node_b.u,
                 node_b.ref_y+scaling_factor*node_b.v,
                 node_b.ref_z+scaling_factor*node_b.w]
            a = [node_a.ref_x+scaling_factor*node_a.u,
                 node_a.ref_y+scaling_factor*node_a.v,
                 node_a.ref_z+scaling_factor*node_a.w]

            lines.append([a, b])

    lc = Line3DCollection(lines, colors=color, linewidths=2)

    ax.add_collection(lc)

    plot_symbols(ax, model, color, initial=False, **options)


def plot_spring(ax, location, direction, **options):
    size = get_max_axes_delta(ax) / 100.0 * options.get('plot/symbol_size', 10)

    n = 1000

    points = np.empty((3, n + 4))

    points[:, 0] = [0, 0, 0]
    points[:, 1] = [5, 0, 0]

    # Plot a helix along the x-axis
    theta_max = 8 * np.pi
    theta = np.linspace(0, theta_max, n)
    points[0, 2:-2] = theta + 4
    points[1, 2:-2] = np.sin(theta) * 3
    points[2, 2:-2] = np.cos(theta) * 3
    points[:, -2] = [theta_max+4, 0, 0]
    points[:, -1] = [theta_max+8, 0, 0]

    points *= size / (8 * np.pi + 8)

    x, y, z = points

    if direction == 'x':
        x, y, z = -x, y, z
    elif direction == 'y':
        x, y, z = y, -x, z
    elif direction == 'z':
        x, y, z = z, y, -x
    else:
        raise RuntimeError()

    x += location[0]
    y += location[1]
    z += location[2]

    ax.plot(x, y, z, 'b', lw=1)


def plot_model(ax, model, color, initial, **options):
    lines = list()

    for element in model.elements:
        if type(element) == Truss:
            node_a = element.node_a
            node_b = element.node_b

            a = [node_a.ref_x, node_a.ref_y, node_a.ref_z] if initial else [node_a.x, node_a.y, node_a.z]
            b = [node_b.ref_x, node_b.ref_y, node_b.ref_z] if initial else [node_b.x, node_b.y, node_b.z]

            lines.append([a, b])
        elif type(element) == Spring:
            location = element.node.location
            if element.kx != 0:
                plot_spring(ax, location, 'x', **options)
            if element.ky != 0:
                plot_spring(ax, location, 'y', **options)
            if element.kz != 0:
                plot_spring(ax, location, 'z', **options)
        else:
            element.plot(initial, lines)

    lc = Line3DCollection(lines, colors=color, linewidths=2)

    ax.add_collection(lc)

    plot_symbols(ax, model, color, initial, **options)


def plot_symbols(ax, model, color, initial, **options):
    if options.get('plot/dirichlet', True):
        plot_boundary_conditions(ax, model, initial, **options)
    if options.get('plot/neumann', True):
        plot_forces(ax, model, initial, **options)
    if options.get('plot/highlight_dof', False):
        plot_dof_higlight(ax, model, initial, **options)


def get_max_axes_delta(ax):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    return max([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]])


def plot_dof_higlight(ax, model, initial, **options):
    size = get_max_axes_delta(ax)/25 * options.get('plot/symbol_size', 5)

    dof = model.dofs[options.get('plot/dof_idx', None)]
    if dof is None:
        return

    node_id, dof_type = dof

    node = model.nodes[node_id]

    dx, dy, dz = 0, 0, 0
    if dof_type == 'u':
        dx = 1
    if dof_type == 'v':
        dy = 1
    if dof_type == 'w':
        dz = 1

    color = 'lightgray' if initial else 'tab:blue'
    if initial:
        x = node.ref_x
        y = node.ref_y
        z = node.ref_z
    else:
        x = node.x
        y = node.y
        z = node.z
    a = get_dof_arrow(x, y, z, dx, dy, dz, size*0.75, color=color)
    ax.add_artist(a)
    # TODO: fix size of spere...
    # a = get_sphere(x, y, z, size/300, color=color)
    # ax.add_artist(a)


def plot_forces(ax, model, initial, **options):
    size = get_max_axes_delta(ax)/25 * options.get('plot/symbol_size', 5)

    for node in model.nodes:
        color = 'lightgray' if initial else 'lightcoral'
        if initial:
            x, y, z = node.ref_location
        else:
            x, y, z = node.location
        a = get_force_arrow(x, y, z, node.fx, node.fy, node.fz, size, color=color)

        if a is not None:
            ax.add_artist(a)


def plot_boundary_conditions(ax, model, initial, **options):
    size = get_max_axes_delta(ax)/100.0 * options.get('plot/symbol_size', 5)

    polygons = list()

    for node in model.nodes:
        for dof_type in ['u', 'v', 'w']:
            if node.dof(dof_type).is_active:
                continue
            if initial:
                polygons.extend(get_tet4_polygons(node.ref_x, node.ref_y, node.ref_z, size, dof_type))
            else:
                polygons.extend(get_tet4_polygons(node.x, node.y, node.z, size, dof_type))

    color = 'lightgray' if initial else 'lightcoral'
    pc = Poly3DCollection(polygons, edgecolor=color, linewidth=0.5, alpha=0.25)
    pc.set_facecolor(color)  # needs to be defined outside otherwhise alpha is not working
    ax.add_collection3d(pc)


def animate_model(fig, ax, models, speed=200, **options):
    bounding_box = get_bounding_box(models)

    def update(step):
        step_model = models[step]

        ax.clear()

        ax.grid()

        plot_bounding_cube(ax, bounding_box)

        ax.set_xlabel('< x >')
        ax.set_ylabel('< y >')
        ax.set_zlabel('< z >')

        title = f'Deformed structure at time step {step}'

        if step_model.name is not None:
            title += f'\n{step_model.name}'

        ax.set_title(title)

        plot_model(ax, step_model, 'gray', True, **options)
        plot_model(ax, step_model, 'red', False, **options)

    return anim.FuncAnimation(fig, update, frames=len(models), repeat=True, interval=speed)


def plot_crosshair(ax, x, y, **kwargs):
    lx = ax.axvline(**kwargs)
    lx.set_xdata(x)

    ly = ax.axhline(**kwargs)
    ly.set_ydata(y)


def plot_custom_curve(ax, *args, **kwargs):
    ax.plot(*args, **kwargs)


def plot_bounding_cube(ax, bounding_box, color='w'):
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box

    xyz_min = np.array([min_x, min_y, min_z])
    xyz_max = np.array([max_x, max_y, max_z])

    max_range = np.array(xyz_max - xyz_min).max()

    center = (xyz_max + xyz_min) / 2

    corners = max_range / 2 * np.mgrid[-1:2:2, -1:2:2, -1:2:2].reshape(3, 8).T + center

    for x, y, z in corners:
        ax.plot([x], [y], [z], color)


def show_animation(model, speed=200, block=True):
    if model.status == ModelStatus.eigenvector:
        return show_eigenvector_animation(model, speed, block)
    else:
        return show_history_animation(model, speed, block)


def show_history_animation(model, speed=200, block=True, size=(10, 10)):
    history = model.get_model_history()

    fig = Figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')

    ani = animate_model(fig, ax, history, speed=speed)

    return HTML(ani.to_jshtml())


def show_eigenvector_animation(model, speed=200, block=True, size=(10, 10)):
    eigenvector = model
    initial_model = model.get_initial_model()

    models = [initial_model, eigenvector]

    bounding_box = get_bounding_box(models)

    fig = Figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')

    def update(step):
        step_model = models[step]

        ax.clear()

        ax.grid()

        plot_bounding_cube(ax, bounding_box)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.title('Eigenvector of Structure \n{}'.format(step_model.name))

        plot_model(ax, step_model, 'gray', True)
        plot_model(ax, step_model, 'red', False)

    ani = anim.FuncAnimation(fig, update, frames=2, repeat=True, interval=speed)

    return HTML(ani.to_jshtml())


def show_deformation_plot(model, step=None, size=(10, 10)):
    bounding_box = get_bounding_box([model])

    fig = Figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')

    ax.clear()

    ax.grid()
    plot_bounding_cube(ax, bounding_box)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plot_model(ax, model, 'gray', True)

    if step is not None:
        model = model.get_model_history()[step]
    else:
        step = len(model.get_model_history())-1

    plot_model(ax, model, 'red', False)

    ax.set_title('Deformed structure at time step {}\n{}'.format(step, model.name))

    fig.show()
