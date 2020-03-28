"""
Non linear example of a simple displacement mechanism
"""

import numpy as np

from nfem import *

#======================================
# Preprocessing
#======================================
model = Model('Mechanism')

model.add_node(id='A', x=0, y=0, z=0)
model.add_node(id='B', x=1, y=0.5, z=0)
model.add_node(id='C', x=2, y=0, z=0, fx=-1)

model.add_truss_element(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
model.add_truss_element(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)

model.add_dirichlet_condition(node_id='A', dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id='B', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='C', dof_types='vw', value=0)

#======================================
# Solve
#======================================

displacement_curve = np.linspace(0.0,-1.0,11)

for displacement in displacement_curve:

    # create a new model for each solution step
    model = model.get_duplicate()

    # prescribe dof -> note that this is a kinematic system:
    # (tangential prediction does not work because of the singular stiffness matrix)
    model.predict_dof_state(value=displacement, dof=('C', 'u') )

    # solve
    model.perform_non_linear_solution_step(strategy="displacement-control", dof=('C', 'u'))

#======================================
# Postprocessing
#======================================

plot = Plot2D()
plot.invert_xaxis()

# plot the load displacement curve using the predefined function
plot.add_load_displacement_curve(model, dof=('C', 'u'))

# plot the first eigenvalue using the general historic plot function
def custom_data_function(model):
    """This function is called for each model in the history. It returns the
    values for x and y of a model.
    """
    x = model[('C', 'u')].delta
    y = model[('B', 'v')].delta
    return x, y
plot.add_history_curve(model,
                        x_y_data=custom_data_function,
                        label='C-u : B-v')

plot.show()

# animated plot
animation = show_animation(model)