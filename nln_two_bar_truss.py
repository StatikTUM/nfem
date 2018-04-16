"""
Non linear example of the two bar truss

It can be run with different path following methods:
1:load control 
2:displacement control
3:arclength control
4:arclength control with delta predictor

This can be set right below
"""
# 1:load control 
# 2:displacement control
# 3:arclength control
# 4:arclength control with delta predictor
method = 4

import numpy as np

from nfem import Model, PlotAnimation, PlotLoadDisplacementCurve
# path following methods
from nfem import LoadControl, DisplacementControl, ArcLengthControl
# predictor methods
from nfem import LoadIncrementPredictor, DisplacementIncrementPredictor, LastIncrementPredictor

# Creation of the model
model = Model('Two-Bar Truss')

model.AddNode(id='A', x=0, y=0, z=0)
model.AddNode(id='B', x=1, y=1, z=0)
model.AddNode(id='C', x=2, y=0, z=0)

model.AddTrussElement(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
model.AddTrussElement(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)

model.AddSingleLoad(id='load 1', node_id='B', fv=-1)

model.AddDirichletCondition(node_id='A', dof_types='uvw', value=0)
model.AddDirichletCondition(node_id='B', dof_types='w', value=0)
model.AddDirichletCondition(node_id='C', dof_types='uvw', value=0)

# Solve with the chosen method
if method == 1: #load control
    # define a load curve with the lambda values that should be used
    load_curve = np.linspace(0.025, 0.5, 20)
    for lam in load_curve:
        # create a new model for each solution step
        model = model.GetDuplicate()

        predictor_method = LoadIncrementPredictor()

        path_following_method = LoadControl(lam)
        
        model.PerformNonLinearSolutionStep(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

elif method == 2: #displacement control
    # define a list of displacement values that should be used
    displacement_curve = np.linspace(-0.1, -2.0, 20)
    for displacement in displacement_curve:
        # create a new model for each solution step
        model = model.GetDuplicate()

        predictor_method = DisplacementIncrementPredictor(dof=('B', 'v'))

        path_following_method = DisplacementControl(dof=('B', 'v'), displacement_hat=displacement)
        
        model.PerformNonLinearSolutionStep(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

elif method == 3: #arclength control
    # define a list of displacement values that should be used
    arclength = 0.12
    n_steps = 20
    for i in range(n_steps):
        # create a new model for each solution step
        model = model.GetDuplicate()

        predictor_method = DisplacementIncrementPredictor(dof=('B', 'v'), value=-1.0)

        path_following_method = ArcLengthControl(l_hat=arclength)
        
        model.PerformNonLinearSolutionStep(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

elif method == 4: #arclength control with delta predictor
    # define a list of displacement values that should be used
    arclength = 0.12
    n_steps = 20
    for i in range(n_steps):
        # create a new model for each solution step
        model = model.GetDuplicate()

        if i == 0:
            predictor_method = DisplacementIncrementPredictor(node_id='B', dof_type='v', value=-1.0)
        else:            
            predictor_method = LastIncrementPredictor()

        path_following_method = ArcLengthControl(l_hat=arclength)
        
        model.PerformNonLinearSolutionStep(predictor_method=predictor_method,
                                           path_following_method=path_following_method)


# get the model history
history = model.GetModelHistory()

# plot the load displacement curve
PlotLoadDisplacementCurve(history, node_id='B', dof_type='v')

# animated plot
PlotAnimation(history)
