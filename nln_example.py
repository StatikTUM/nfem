"""
Here, multiple two bar trusses are simulated at the same time to test efficiency
"""

import numpy as np

from nfem import Model, PlotAnimation, PlotLoadDisplacementCurve
# path following methods
from nfem import LoadControl, DisplacementControl, ArcLengthControl
# predictor methods
from nfem import LoadIncrementPredictor, DisplacementIncrementPredictor

# Number of two bar trusses
n_models = 1

# Number of two bar steps
n_steps = 1

# Creating two bar trusses in a loop
model = Model('Initial Model')
node_count = 0
element_count = 0

for i in range(n_models):
    id_offset = 100 * i

    z = i * 0.1

    node_a = id_offset + 1
    node_b = id_offset + 2
    node_c = id_offset + 3

    model.AddNode(id=node_a, x=0, y=0, z=z)
    model.AddNode(id=node_b, x=1, y=1, z=z)
    model.AddNode(id=node_c, x=2, y=0, z=z)

    truss_1 = id_offset + 11
    truss_2 = id_offset + 12

    model.AddTrussElement(id=truss_1, node_a=node_a, node_b=node_b, youngs_modulus=1, area=1)
    model.AddTrussElement(id=truss_2, node_a=node_b, node_b=node_c, youngs_modulus=1, area=1)

    load_b = id_offset + 21

    model.AddSingleLoad(id=load_b, node_id=node_b, fv=-1)

    model.AddDirichletCondition(node_id=node_a, dof_types='uvw', value=0)
    model.AddDirichletCondition(node_id=node_b, dof_types='w', value=0)
    model.AddDirichletCondition(node_id=node_c, dof_types='uvw', value=0)

# 1:load control 
# 2:displacement control
# 3:arclength control
method = 3

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

        predictor_method = DisplacementIncrementPredictor(node_id=2, dof_type='v')

        path_following_method = DisplacementControl(node_id=2, dof_type='v', displacement_hat=displacement)
        
        model.PerformNonLinearSolutionStep(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

elif method == 3: #arclength control
    # define a list of displacement values that should be used
    arclength = 0.12
    n_steps = 20
    for i in range(n_steps):
        # create a new model for each solution step
        model = model.GetDuplicate()

        predictor_method = DisplacementIncrementPredictor(node_id=2, dof_type='v', value=-1.0)

        path_following_method = ArcLengthControl(l_hat=arclength)
        
        model.PerformNonLinearSolutionStep(predictor_method=predictor_method,
                                           path_following_method=path_following_method)


# get the model history
history = model.GetModelHistory()

# plot the load displacement curve
PlotLoadDisplacementCurve(history, node_id=2, dof_type='v')
# animated plot
PlotAnimation(history)
