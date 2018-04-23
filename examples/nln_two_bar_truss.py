"""
Non linear example of the two bar truss

It can be run with different path following methods:
1: load control 
2: displacement control
3: arclength control
4: arclength control with delta predictor

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
# Solve (with the chosen method)
# 1: load control 
# 2: displacement control
# 3: arclength control
# 4: arclength control with delta predictor
#======================================
method = 4

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
    arclength = 0.1
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
    arclength = 0.1
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

#======================================
# Postprocessing
#======================================

plot = Plot2D()
# plot the load displacement curve
plot.add_history_curve(model, 
                        x_data=lambda model: model.get_dof_state(dof=('B', 'v')), 
                        y_data=lambda model: model.lam,
                        label='$\lambda$ : B-v')
# plot det(K) 
plot.add_history_curve(model, 
                        x_data=lambda model: model.get_dof_state(dof=('B', 'v')), 
                        y_data=lambda model: model.det_k,
                        label='det(K) : B-v')
# plot a custom curve
x=[1,-1]
y=[0, 1]
plot.add_custom_curve(x, y, label="my_custom_curve", linewidth=5)
plot.show()

# animated plot
animation = Animation3D()
animation.show(model)

# static plot
deformation_plot = DeformationPlot3D()
deformation_plot.show(model)