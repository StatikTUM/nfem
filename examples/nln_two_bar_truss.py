"""
Non linear example of the two bar truss

It can be run with different path following and predictor methods:
This can be set right below in the 'Solve' block
"""
# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
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
method = 3

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
        
        model.perform_non_linear_solution_step(strategy="load-control")

elif method == 2: #displacement control

    # define a list of displacement values that should be used
    displacement_curve = np.linspace(-0.1, -2.0, 20)
    for displacement in displacement_curve:

        # create a new model for each solution step
        model = model.get_duplicate()

        # prescribe u
        model.predict_tangential(strategy="dof", value=displacement, dof=('B', 'v') )
        # ALTERNATIVE prescribe delta u
        #model.predict_tangential(strategy="delta-u", value=-0.1, dof=('B', 'v') )
        
        model.perform_non_linear_solution_step(strategy="displacement-control", dof=('B', 'v'))

elif method == 3: #arclength control

    # define a list of displacement values that should be used
    displacement_increment = -0.1
    n_steps = 20
    for i in range(n_steps):

        # create a new model for each solution step
        model = model.get_duplicate()

        # prescribe delta_u
        model.predict_tangential(strategy="delta-dof", value=displacement_increment, dof=('B', 'v') )
        
        model.perform_non_linear_solution_step(strategy="arc-length-control")

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
            model.predict_tangential(strategy="delta-u", value=-0.1, dof=('B', 'v') )
        else:
            # increment dof and lambda with the increment from the last solution step
            model.predict_with_last_increment()
        
        model.perform_non_linear_solution_step(strategy="arc-length-control")

#======================================
# Postprocessing
#======================================

# plot the load displacement curve
plot = Plot2D()
plot.add_load_displacement_curve(model, dof=('B', 'v'))
plot.add_load_displacement_curve(model, dof=('B', 'u'))
plot.show()

# animated plot
animation = Animation3D()
animation.show(model)

# static plot
deformation_plot = DeformationPlot3D()
deformation_plot.show(model)