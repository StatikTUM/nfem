import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from ..truss import Truss

def BoundingBox(model):
    nodes = [node for model in model.GetModelHistory() for node in model.nodes.values()]

    min_x = min(node.x for node in nodes)
    max_x = max(node.x for node in nodes)

    min_y = min(node.y for node in nodes)
    max_y = max(node.y for node in nodes)

    min_z = min(node.z for node in nodes)
    max_z = max(node.z for node in nodes)

    return min_x, max_x, min_y, max_y, min_z, max_z

def PlotModel(ax, model, color, initial):
    xys = list()
    zs = list()

    for element in model.elements.values():
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

def PlotLoadDisplacementCurve(ax, model, dof):
    history = model.GetModelHistory()

    x_data = np.zeros(len(history))
    y_data = np.zeros(len(history))

    node_id, dof_type = dof

    for i, model in enumerate(history):
        x_data[i] = model.GetDofState(dof)
        y_data[i] = model.lam

    ax.plot(x_data, y_data, '-o')

def ShowLoadDisplacementCurve(model, dof, switch_x_axis=True):
    dof_type, node_id = dof

    fig, ax = plt.subplots()

    ax.set(xlabel='{} at node {}'.format(dof_type, node_id),
           ylabel='Load factor ($\lambda$)',
           title='Load-displacement diagram')
    ax.set_facecolor('white')
    ax.grid()

    PlotLoadDisplacementCurve(ax, model, dof)

    if switch_x_axis:
        plt.gca().invert_xaxis()

    plt.show()

def ShowHistoryAnimation(model, speed=200):
    history = model.GetModelHistory()

    min_x, max_x, min_y, max_y, min_z, max_z = BoundingBox(model)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def Update(step):
        step_model = history[step]

        ax.clear()

        ax.grid()

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.title('Deformed structure at time step {}\n{}'.format(step, step_model.name))

        PlotModel(ax, step_model, 'gray', True)
        PlotModel(ax, step_model, 'red', False)

    a = anim.FuncAnimation(fig, Update, frames=len(history), repeat=True, interval=speed)

    plt.show()