import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
import numpy as np

from .model import Truss

def PlotAnimation(history, speed=200):
    history_size = len(history)

    max_x, max_y, min_x, min_y, max_z, min_z = FindLimits(history)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def Update(step):
        ax.clear()
        step_model = history[step]
        for element in step_model.elements.values():
            # Truss elements
            if type(element) == Truss:
                node_a = element.node_a
                node_b = element.node_b
                # TODO LineCollection for speedup
                ax.plot([node_a.x, node_b.x], [node_a.y, node_b.y], [node_a.z, node_b.z], color='blue')

        ax.grid()
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)
        ax.set_zlim(min_z,max_z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('Deformed structure at time step {}\n{}'.format(step, step_model.name))

    a = anim.FuncAnimation(fig, Update, frames=history_size, repeat=True, interval=speed)
    plt.show()

def FindLimits(history):
    """Finds the bounding box for all models in the history."""

    nodes = [node for model in history for node in model.nodes.values()]

    min_x = min(node.x for node in nodes)
    max_x = max(node.x for node in nodes)

    min_y = min(node.y for node in nodes)
    max_y = max(node.y for node in nodes)

    min_z = min(node.z for node in nodes)
    max_z = max(node.z for node in nodes)

    return max_x, max_y, min_x, min_y, max_z, min_z

def PlotLoadDisplacementCurve(history, node_id, dof_type, switch_x_axis=True):
    x_data = np.zeros(len(history))
    y_data = np.zeros(len(history))

    dof = (node_id, dof_type)
    # Data for plotting
    initial_model = history[0]
    for i, model in enumerate(history):
        x_data[i] = model.GetDofState(dof)
        y_data[i] = model.lam

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure() and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    plotted_line = ax.plot(x_data, y_data, '-o')

    ax.legend((plotted_line), ('Node '+ str(node_id)+' - Dof '+dof_type,), loc='upper left')

    ax.set(xlabel='u', ylabel='lambda',
        title='Load displacement diagram')
    ax.grid()

    plt.gca().invert_xaxis()

    plt.show()
