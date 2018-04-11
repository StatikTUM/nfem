from nfem import Model, Assembler, History, plotting_utility
import numpy as np

# Here, multiple two bar trusses are simulated at the same time to test efficiency

# Number of two bar trusses
n_models = 100

# Number of two bar steps
n_steps = 100

# Creating two bar trusses in a loop
model = Model('Initial Model')
node_count = 0
element_count = 0
for i in range(0,n_models):
    z = i*0.1
    node_a = node_count
    model.AddNode(id=node_a, x= 0, y=0, z=z)
    node_count += 1
    node_b = node_count
    model.AddNode(id=node_b, x= 5, y=2, z=z)
    node_count += 1
    node_c = node_count
    model.AddNode(id=node_c, x=10, y=0, z=z)
    node_count += 1
    model.AddTrussElement(id=element_count, node_a=node_a, node_b=node_b, youngs_modulus=10, area=2)
    element_count += 1
    model.AddTrussElement(id=element_count, node_a=node_b, node_b=node_c, youngs_modulus=10, area=2)
    element_count += 1
    model.AddDirichletCondition(node_id=node_a, dof_types='uvw', value=0)
    model.AddDirichletCondition(node_id=node_b, dof_types='w'  , value=0)
    model.AddDirichletCondition(node_id=node_c, dof_types='uvw', value=0)
    model.AddSingleLoad(id=element_count, node_id=node_b, fv=-1)
    element_count += 1

# initializing history
history = History(model)
lam = 0.1

# solving a linear system in each step
for step in range(1,n_steps):
    model.PerformLinearSolutionStep(lam*step)
    history.AddModel(step, model)

# print the result of one step
print('Deformed step',n_steps,':')
deformed = history.GetModel(n_steps-1)
print(deformed.nodes[1].x)
print(deformed.nodes[1].y)
print(deformed.nodes[1].z)

# animated plot
plotting_utility.plot_cont_animated(history,speed=1)
