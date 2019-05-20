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

model.add_node(id='A', x=0, y=0, z=0)
model.add_node(id='B', x=1, y=1, z=0)
model.add_node(id='C', x=2, y=0, z=0)

model.add_truss_element(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
model.add_truss_element(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)

model.add_single_load(id='load 1', node_id='B', fv=-1)

model.add_dirichlet_condition(node_id='A', dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id='B', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='C', dof_types='uvw', value=0)

load_curve = np.linspace(0.025, 0.5, 20)
for lam in load_curve:
    # create a new model for each solution step
    model = model.get_duplicate()    
    model.lam = lam
    model.perform_linear_solution_step()

# get the model history
history = model.get_model_history()

# plot the load displacement curve
show_load_displacement_curve(model, dof = ('B', 'v'))

# animated plot
show_animation(model)
