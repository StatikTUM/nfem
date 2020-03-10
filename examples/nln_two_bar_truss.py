"""
Non linear example of the two bar truss

It can be run with different path following and predictor methods:
This can be set right below in the 'Solve' block
"""

import numpy as np

from nfem import *

#======================================
# Preprocessing
#======================================
model = Model('Two-Bar Truss')

model.add_node(id='A', x=0, y=0, z=0)
model.add_node(id='B', x=1, y=1, z=0)
model.add_node(id='C', x=2, y=0, z=0)

model.add_truss_element(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
model.add_truss_element(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)

model.add_single_load(id='load 1', node_id='B', fv=-1)

model.add_dirichlet_condition(node_id='A', dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id='B', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='C', dof_types='uvw', value=0)

#======================================
# Solve
# (with the chosen method)
#======================================
# 1: load control with tangential 'lambda' predictor
# 2: displacement control with tangential 'u' predictor
# 3: arclength control with tangential 'delta-u' predictor
# 4: arclength control with tangential 'arc-lenght' predictor
# 5: arclength control with increment predictor
#======================================
method = 5

if method == 1: #load control

    # define a load curve with the lambda values that should be used
    load_curve = np.linspace(0.025, 0.5, 20)
    for lam in load_curve:

        # create a new model for each solution step
        model = model.get_duplicate()

        # prescribe lambda
        model.predict_tangential(strategy="lambda", value=lam)
        # ALTERNATIVE prescribe delta lambda
        #model.predict_tangential(strategy="delta-lambda", value=0.025)

        model.perform_non_linear_solution_step(strategy="load-control",
                                               solve_det_k=True,
                                               solve_attendant_eigenvalue=True)

elif method == 2: #displacement control

    # define a list of displacement values that should be used
    displacement_curve = np.linspace(-0.1, -2.0, 20)
    for displacement in displacement_curve:

        # create a new model for each solution step
        model = model.get_duplicate()

        # prescribe dof
        model.predict_tangential(strategy="dof", value=displacement, dof=('B', 'v') )
        # ALTERNATIVE prescribe delta dof
        #model.predict_tangential(strategy="delta-dof", value=-0.1, dof=('B', 'v') )

        model.perform_non_linear_solution_step(strategy="displacement-control", dof=('B', 'v'),
                                               solve_det_k=True,
                                               solve_attendant_eigenvalue=True)

elif method == 3: #arclength control

    # define a list of displacement values that should be used
    displacement_increment = -0.1
    n_steps = 20
    for i in range(n_steps):

        # create a new model for each solution step
        model = model.get_duplicate()

        # prescribe delta dof
        model.predict_tangential(strategy="delta-dof", value=displacement_increment, dof=('B', 'v') )

        model.perform_non_linear_solution_step(strategy="arc-length-control",
                                               solve_det_k=True,
                                               solve_attendant_eigenvalue=True)

elif method == 4: #arclength control with delta predictor

    # define a list of displacement values that should be used
    arclength = 0.1
    n_steps = 20
    for i in range(n_steps):

        # create a new model for each solution step
        model = model.get_duplicate()

        if i == 0:
            # increment the dof state
            model.predict_tangential(strategy="delta-dof", value=-0.1, dof=('B', 'v') )
        else:
            # increment dof and lambda with the increment from the last solution step
            model.predict_tangential(strategy="arc-length")

        model.perform_non_linear_solution_step(strategy="arc-length-control")

elif method == 5: #arclength control with delta predictor

    # define a list of displacement values that should be used
    arclength = 0.1
    n_steps = 20
    for i in range(n_steps):

        # create a new model for each solution step
        model = model.get_duplicate()

        if i == 0:
            # increment the dof state
            model.predict_tangential(strategy="delta-dof", value=-0.1, dof=('B', 'v') )
        else:
            # increment dof and lambda with the increment from the last solution step
            model.predict_with_last_increment()

        model.perform_non_linear_solution_step(strategy="arc-length-control",
                                               solve_det_k=True,
                                               solve_attendant_eigenvalue=True)

#======================================
# Postprocessing
#======================================

plot = Plot2D()
plot.invert_xaxis()

# plot the load displacement curve using the predefined function
plot.add_load_displacement_curve(model, dof=('B', 'v'), show_iterations=True)

# plot det(K) using the general historic plot function
def det_k_data_function(model):
    """This function is called for each model in the history. It returns the
    values for x and y of a model.
    """
    x = model.get_dof_state(dof=('B', 'v'))
    y = model.det_k
    return x, y
plot.add_history_curve(model,
                        x_y_data=det_k_data_function,
                        label='det(K) : B-v')

# plot the first eigenvalue using the general historic plot function
def eigenvalue_data_function(model):
    """This function is called for each model in the history. It returns the
    values for x and y of a model.
    """
    x = model.get_dof_state(dof=('B', 'v'))
    if model.first_eigenvalue is None:
        y = None
    else:
        y = model.first_eigenvalue * model.lam
    return x, y
plot.add_history_curve(model,
                        x_y_data=eigenvalue_data_function,
                        label='eigval * lambda : B-v')

# plot a custom curve using matplotlib syntax
x=[0,-2]
y=[0, 0]
plot.add_custom_curve(x, y, label="my_custom_curve", linewidth=0.5)
plot.show()

# animated plot
animation = show_animation(model)

# static plot
show_deformation_plot(model)