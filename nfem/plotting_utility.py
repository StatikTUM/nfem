import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim

from .model import Truss

def PlotAnimation(history, speed=1):
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
        plt.title(f'Deformed structure at time step {step}\n{step_model.name}')

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
