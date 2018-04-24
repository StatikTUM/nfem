"""
Here, multiple two bar trusses are simulated at the same time to test efficiency
"""
# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
import numpy as np

from nfem import *

# Number of two bar trusses
n_models = 100

# Number of two bar steps
n_steps = 100

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

    model.add_node(id=node_a, x=0, y=0, z=z)
    model.add_node(id=node_b, x=5, y=2, z=z)
    model.add_node(id=node_c, x=10, y=0, z=z)

    truss_1 = id_offset + 11
    truss_2 = id_offset + 12

    model.add_truss_element(id=truss_1, node_a=node_a, node_b=node_b, youngs_modulus=10, area=2)
    model.add_truss_element(id=truss_2, node_a=node_b, node_b=node_c, youngs_modulus=10, area=2)

    load_b = id_offset + 21

    model.add_single_load(id=load_b, node_id=node_b, fv=-1)

    model.add_dirichlet_condition(node_id=node_a, dof_types='uvw', value=0)
    model.add_dirichlet_condition(node_id=node_b, dof_types='w', value=0)
    model.add_dirichlet_condition(node_id=node_c, dof_types='uvw', value=0)

# solving a linear system in each step
for lam in np.linspace(0, 10, n_steps+1):
    model = model.get_duplicate()
    model.name = 'lambda = ' + str(lam)
    model.lam = lam
    model.perform_linear_solution_step()

history = model.get_model_history()

# print the result of last step
deformed = history[-1]
print('Deformed step {}:'.format(n_steps))
print(deformed.get_node(id=2).x)
print(deformed.get_node(id=2).y)
print(deformed.get_node(id=2).z)

# animated plot
show_history_animation(model)
