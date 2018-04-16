"""
Linear example of the two bar truss
"""
# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
import numpy as np

from nfem import *

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

load_curve = np.linspace(0.025, 0.5, 20)
for lam in load_curve:
    # create a new model for each solution step
    model = model.GetDuplicate()    
    model.lam = lam
    model.PerformLinearSolutionStep()

# get the model history
history = model.GetModelHistory()

# plot the load displacement curve
ShowLoadDisplacementCurve(model, dof = ('B', 'v'))

# animated plot
ShowHistoryAnimation(model)
