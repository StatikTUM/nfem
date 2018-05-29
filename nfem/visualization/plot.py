"""This module contains helpers for visualize data.

Authors: Klaus Sautter, Thomas Oberbichler, Armin Geiser
"""

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from ..truss import Truss
from ..model import ModelStatus

class Plot2D(object):

    def __init__(self, x_label='Displacement', y_label='Load factor ($\lambda$)',
                 title='Load-displacement diagram'):
        self.fig, self.ax = plt.subplots()

        self.ax.set(xlabel=x_label,
            ylabel=y_label,
            title=title)
        self.ax.set_facecolor('white')
        self.ax.grid()

        self.legend = []

    def invert_xaxis(self):
        """Invert the x axis of the plot"""
        self.ax.invert_xaxis()

    def add_load_displacement_curve(self, model, dof, label=None, show_iterations=False):
        plot_load_displacement_curve(self.ax, model, dof, label)
        if show_iterations:
            plot_load_displacement_iterations(self.ax, model, dof, label)

    def add_det_k_curve(self, model, dof, label=None):
        plot_det_k_curve(self.ax, model, dof, label=label)

    def add_history_curve(self, model, x_y_data, fmt='-o', **kwargs):
        """Add a history curve of the model to the plot.

        Parameters
        ----------
        model : Model
            Model object of which the history will be printed
        x_y_data : function(Model) that returns the value for the x and y axis
            of a models state. It is called for all models in the history
        fmt: matplotlib format string e.g. '-o' for line with points
            For details visit the link below
        **kwargs: additional format arguments e.g. label="My label" to give the
            curve a name.
            for details visit the link below
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
        """
        plot_history_curve(self.ax, model, x_y_data, fmt, **kwargs)

    def add_custom_curve(self, *args, **kwargs):
        """Add a custom curve to the plot.
        Uses the syntax of the matplotlip.pyplot.plot function

        For a description of the possible parameters visit
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
        """
        plot_custom_curve(self.ax, *args, **kwargs)

    def show(self):
        """Shows the plot with all the curves that have been added.
        """
        self.ax.legend(loc='best')
        plt.show()

class Animation3D(object):

    def show(self, model, speed=200):
        if model.status == ModelStatus.eigenvector:
            self.animation = show_eigenvector_animation(model, speed)
        else:
            self.animation = show_history_animation(model, speed)

class DeformationPlot3D(object):

    def show(self, model, step=None):
        show_deformation_plot(model, step)


def get_bounding_box(models):
    nodes = [node for model in models for node in model.nodes]

    min_x = min(node.x for node in nodes)
    max_x = max(node.x for node in nodes)

    min_y = min(node.y for node in nodes)
    max_y = max(node.y for node in nodes)

    min_z = min(node.z for node in nodes)
    max_z = max(node.z for node in nodes)

    return min_x, max_x, min_y, max_y, min_z, max_z

def plot_model(ax, model, color, initial):
    lines = list()

    for element in model.elements:
        if type(element) == Truss:
            node_a = element.node_a
            node_b = element.node_b

            a = [node_a.reference_x, node_a.reference_y, node_a.reference_z] if initial else [node_a.x, node_a.y, node_a.z]
            b = [node_b.reference_x, node_b.reference_y, node_b.reference_z] if initial else [node_b.x, node_b.y, node_b.z]

            lines.append([a, b])

    lc = Line3DCollection(lines, colors=color, linewidths=2)

    ax.add_collection(lc)

def animate_model(fig, ax, models, speed=200):

    bounding_box = get_bounding_box(models)

    def update(step):
        step_model = models[step]

        ax.clear()

        ax.grid()

        plot_bounding_cube(ax, bounding_box)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_title('Deformed structure at time step {}\n{}'.format(step, step_model.name))

        plot_model(ax, step_model, 'gray', True)
        plot_model(ax, step_model, 'red', False)

    a = anim.FuncAnimation(fig, update, frames=len(models), repeat=True, interval=speed)

    return a

def plot_load_displacement_iterations(ax, model, dof, label=None):
    history = model.get_model_history(skip_iterations=False)

    x_data = np.zeros(len(history))
    y_data = np.zeros(len(history))

    node_id, dof_type = dof

    for i, model in enumerate(history):
        x_data[i] = model.get_dof_state(dof)
        y_data[i] = model.lam

    if label == None:
        label = '$\lambda$ : {} at node {} (iter)'.format(dof_type, node_id)
    else:
        label += ' (iter)'
    ax.plot(x_data, y_data, '--o', linewidth=0.75, markersize=2.0, label=label)


def plot_load_displacement_curve(ax, model, dof, label=None):
    history = model.get_model_history()

    x_data = np.zeros(len(history))
    y_data = np.zeros(len(history))

    node_id, dof_type = dof

    for i, model in enumerate(history):
        x_data[i] = model.get_dof_state(dof)
        y_data[i] = model.lam

    if label is None:
        label = '$\lambda$ : {} at node {}'.format(dof_type, node_id)
    ax.plot(x_data, y_data, '-o', label=label)

def plot_det_k_curve(ax, model, dof, label=None):
    history = model.get_model_history()

    x_data = np.zeros(len(history))
    y_data = np.zeros(len(history))

    node_id, dof_type = dof

    for i, model in enumerate(history):
        x_data[i] = model.get_dof_state(dof)
        y_data[i] = model.det_k

    if label is None:
        label = 'det(K) : {} at node {}'.format(dof_type, node_id)
    ax.plot(x_data, y_data, '-o', label=label)

def plot_history_curve(ax, model, xy_function, fmt, **kwargs):
    history = model.get_model_history()

    x_data = np.zeros(len(history))
    y_data = np.zeros(len(history))

    for i, model in enumerate(history):
        x_data[i], y_data[i]= xy_function(model)

    ax.plot(x_data, y_data, fmt, **kwargs)

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

def show_load_displacement_curve(model, dof, invert_xaxis=True):
    dof_type, node_id = dof

    fig, ax = plt.subplots()

    ax.set(xlabel='{} at node {}'.format(dof_type, node_id),
           ylabel='Load factor ($\lambda$)',
           title='Load-displacement diagram')
    ax.set_facecolor('white')
    ax.grid()

    plot_load_displacement_curve(ax, model, dof)

    if invert_xaxis:
        ax.invert_xaxis()

    plt.show()

def show_history_animation(model, speed=200):
    history = model.get_model_history()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    a = animate_model(fig, ax, history, speed=speed)

    plt.show()

    return a

def show_eigenvector_animation(model, speed=200):
    eigenvector = model
    initial_model = model.get_initial_model()

    models = [initial_model, eigenvector]

    bounding_box = get_bounding_box(models)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(step):
        step_model = models[step]

        ax.clear()

        ax.grid()

        plot_bounding_cube(ax, bounding_box)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.title('Eigenvector of Structure \n{}'.format(step_model.name))

        plot_model(ax, step_model, 'gray', True)
        plot_model(ax, step_model, 'red', False)

    a = anim.FuncAnimation(fig, update, frames=2, repeat=True, interval=speed)

    plt.show()

    return a

def show_deformation_plot(model, step=None):

    bounding_box = get_bounding_box([model])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.clear()

    ax.grid()
    plot_bounding_cube(ax, bounding_box)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plot_model(ax, model, 'gray', True)

    if step != None:
        model = model.get_model_history()[step]
    else:
        step = len(model.get_model_history())-1

    plot_model(ax, model, 'red', False)

    ax.set_title('Deformed structure at time step {}\n{}'.format(step, model.name))

    plt.show()
