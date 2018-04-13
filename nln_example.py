"""
Here, multiple two bar trusses are simulated at the same time to test efficiency
"""

import numpy as np

from nfem import Model, PlotAnimation

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

# solving a linear system in each step
model.PerformNonLinearSolutionStep(prescribed_value=0.1)
exit()

history = model.GetModelHistory()

# print the result of last step
deformed = history[-1]
print(f'Deformed step {n_steps}:')
print(deformed.nodes[2].x)
print(deformed.nodes[2].y)
print(deformed.nodes[2].z)

# animated plot
PlotAnimation(history)
