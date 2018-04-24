"""This module contains helpers for visualize data.

Authors: Klaus Sautter, Thomas Oberbichler, Armin Geiser
"""

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from ..truss import Truss

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

    def add_load_displacement_curve(self, model, dof):
        plot_load_displacement_curve(self.ax, model, dof)

    def show(self):
        self.ax.legend(loc='upper left') 
        plt.show()

class Animation3D(object):

    def show(self, model, speed=200):
        self.animation = show_history_animation(model, speed)

class DeformationPlot3D(object):

    def show(self, model, step=None):
        show_deformation_plot(model, step)


def bounding_box(model):
    nodes = [node for model in model.get_model_history() for node in model.nodes]

    min_x = min(node.x for node in nodes)
    max_x = max(node.x for node in nodes)

    min_y = min(node.y for node in nodes)
    max_y = max(node.y for node in nodes)

    min_z = min(node.z for node in nodes)
    max_z = max(node.z for node in nodes)

    return min_x, max_x, min_y, max_y, min_z, max_z

def plot_model(ax, model, color, initial):
    xys = list()
    zs = list()

    for element in model.elements:
        if type(element) == Truss:
            node_a = element.node_a
            node_b = element.node_b

            a = (node_a.reference_x, node_a.reference_y) if initial else (node_a.x, node_a.y)
            b = (node_b.reference_x, node_b.reference_y) if initial else (node_b.x, node_b.y)
            z = node_b.reference_z if initial else node_b.z

            xys.append([a, b])
            zs.append(z)

    lc = LineCollection(xys, colors=color, linewidths=2)

    ax.add_collection3d(lc, zs=zs)

def plot_load_displacement_curve(ax, model, dof):
    history = model.get_model_history()

    x_data = np.zeros(len(history))
    y_data = np.zeros(len(history))

    node_id, dof_type = dof

    for i, model in enumerate(history):
        x_data[i] = model.get_dof_state(dof)
        y_data[i] = model.lam

    label = '{} at node {}'.format(dof_type, node_id)
    ax.plot(x_data, y_data, '-o', label=label)

def plot_bounding_cube(ax, model, color='w'):
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box(model)

    xyz_min = np.array([min_x, min_y, min_z])
    xyz_max = np.array([max_x, max_y, max_z])

    max_range = np.array(xyz_max - xyz_min).max()

    center = (xyz_max + xyz_min) / 2

    corners = max_range / 2 * np.mgrid[-1:2:2, -1:2:2, -1:2:2].reshape(3, 8).T + center

    for x, y, z in corners:
        ax.plot([x], [y], [z], color)

def show_load_displacement_curve(model, dof, switch_x_axis=True):
    dof_type, node_id = dof

    fig, ax = plt.subplots()

    ax.set(xlabel='{} at node {}'.format(dof_type, node_id),
           ylabel='Load factor ($\lambda$)',
           title='Load-displacement diagram')
    ax.set_facecolor('white')
    ax.grid()

    plot_load_displacement_curve(ax, model, dof)

    if switch_x_axis:
        plt.gca().invert_xaxis()

    plt.show()

def show_history_animation(model, speed=200):
    history = model.get_model_history()

    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box(model)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(step):
        step_model = history[step]

        ax.clear()

        ax.grid()

        plot_bounding_cube(ax, model) 

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.title('Deformed structure at time step {}\n{}'.format(step, step_model.name))

        plot_model(ax, step_model, 'gray', True)
        plot_model(ax, step_model, 'red', False)

    a = anim.FuncAnimation(fig, update, frames=len(history), repeat=True, interval=speed)

    plt.show()

    return a

def show_deformation_plot(model, step=None):

    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box(model)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.clear()

    ax.grid()
    plot_bounding_cube(ax, model) 

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plot_model(ax, model, 'gray', True)

    if step != None:
        model = model.get_model_history()[step]
    else:
        step = len(model.get_model_history())-1

    plot_model(ax, model, 'red', False)
        
    plt.title('Deformed structure at time step {}\n{}'.format(step, model.name))

    plt.show()
