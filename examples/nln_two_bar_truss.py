"""
Non linear example of the two bar truss

It can be run with different path following methods:
1:load control 
2:displacement control
3:arclength control
4:arclength control with delta predictor

This can be set right below
"""
# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
import numpy as np

from nfem import *

# 1:load control 
# 2:displacement control
# 3:arclength control
# 4:arclength control with delta predictor
method = 4

# Creation of the model
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

# Solve with the chosen method
if method == 1: #load control
    # define a load curve with the lambda values that should be used
    load_curve = np.linspace(0.025, 0.5, 20)
    for lam in load_curve:
        # create a new model for each solution step
        model = model.get_duplicate()

        predictor_method = LoadIncrementPredictor()

        path_following_method = LoadControl(lam)
        
        model.perform_non_linear_solution_step(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

elif method == 2: #displacement control
    # define a list of displacement values that should be used
    displacement_curve = np.linspace(-0.1, -2.0, 20)
    for displacement in displacement_curve:
        # create a new model for each solution step
        model = model.get_duplicate()

        predictor_method = DisplacementIncrementPredictor(dof=('B', 'v'))

        path_following_method = DisplacementControl(dof=('B', 'v'), displacement_hat=displacement)
        
        model.perform_non_linear_solution_step(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

elif method == 3: #arclength control
    # define a list of displacement values that should be used
    arclength = 0.12
    n_steps = 20
    for i in range(n_steps):
        # create a new model for each solution step
        model = model.get_duplicate()

        predictor_method = DisplacementIncrementPredictor(dof=('B', 'v'), value=-1.0)

        path_following_method = ArcLengthControl(l_hat=arclength)
        
        model.perform_non_linear_solution_step(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

elif method == 4: #arclength control with delta predictor
    # define a list of displacement values that should be used
    arclength = 0.12
    n_steps = 20
    for i in range(n_steps):
        # create a new model for each solution step
        model = model.get_duplicate()

        if i == 0:
            predictor_method = DisplacementIncrementPredictor(dof=('B', 'v'), value=-1.0)
        else:
            predictor_method = LastIncrementPredictor()

        path_following_method = ArcLengthControl(l_hat=arclength)
        
        model.perform_non_linear_solution_step(predictor_method=predictor_method,
                                           path_following_method=path_following_method)


# get the model history
history = model.get_model_history()

# plot the load displacement curve
plot = Plot2D()
plot.add_load_displacement_curve(model, dof=('B', 'v'))
plot.add_load_displacement_curve(model, dof=('B', 'u'))
plot.show()

# animated plot
show_history_animation(model)
