import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim

from .model import Truss

def plot_cont_animated(history,speed = 1):

    history_size = history.ReturnHistorySize()
    max_x,max_y,min_x,min_y,max_z,min_z = find_max_min_entry(history)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')


    def update(step):
        ax.clear()
        step_model = history.GetModel(step)
        for key, value in step_model.elements.items():
            # Truss elements
            if type(value) == Truss:
                node_a = value.node_a
                node_b = value.node_b
                # TODO LineCollection for speedup
                ax.plot([node_a.x,node_b.x], [node_a.y,node_b.y], [node_a.z,node_b.z], color ='blue')

        ax.grid()
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)
        ax.set_zlim(min_z,max_z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('Deformed structure at time step %i' %step)



    a = anim.FuncAnimation(fig, update, frames=history_size, repeat=True, interval = speed)
    plt.show()


def find_max_min_entry(history):
    ## find max entry for limits
    max_x,max_y,min_x,min_y,max_z,min_z = 0.0,0.0,0.0,0.0,0.0,0.0

    history_size = history.ReturnHistorySize()
    for step in range(history_size):
        step_model = history.GetModel(step)
        for key, value in step_model.nodes.items():
            x_i = value.x
            y_i = value.y
            z_i = value.z
            if x_i < min_x:
                min_x = x_i
            elif x_i > max_x:
                max_x = x_i

            if y_i < min_y:
                min_y = y_i
            elif y_i > max_y:
                max_y = y_i

            if z_i < min_z:
                min_z = z_i
            elif z_i > max_z:
                max_z = z_i

    return max_x,max_y,min_x,min_y,max_z,min_z