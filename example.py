"""
A basic example for Tutorial 1
"""

import numpy as np

from nfem import Model, PlotAnimation

model = Model('Initial Model')
model.AddNode(id='A', x=0, y=0, z=0)
model.AddNode(id='B', x=5, y=2, z=0)
model.AddNode(id='C', x=10, y=0, z=0)
model.AddTrussElement(id=1, node_a='A', node_b='B', youngs_modulus=10, area=2)
model.AddTrussElement(id=2, node_a='B', node_b='C', youngs_modulus=10, area=2)
model.AddDirichletCondition(node_id='A', dof_types='uvw', value=0)
model.AddDirichletCondition(node_id='B', dof_types='w', value=0)
model.AddDirichletCondition(node_id='C', dof_types='uvw', value=0)
model.AddSingleLoad(id='F', node_id='B', fv=-1)

n_steps = 10

for lam in np.linspace(0, 10, n_steps+1):
    model = model.GetDuplicate()
    model.name = 'lambda = ' + str(lam)
    model.lam = lam
    model.PerformLinearSolutionStep()

initial = model.GetInitialModel()

print(initial)

print('Initial:')
print(initial.nodes['B'].x)
print(initial.nodes['B'].y)
print(initial.nodes['B'].z)

history = model.GetModelHistory()

for step, deformed in enumerate(history):
    print('Deformed step {}:'.format(step))
    print(deformed.nodes['B'].x)
    print(deformed.nodes['B'].y)
    print(deformed.nodes['B'].z)

PlotAnimation(history, speed=500)
