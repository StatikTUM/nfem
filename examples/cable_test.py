"""truss cantilever with with cables as diagonals""" 

# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
import numpy as np

from nfem import *

#======================================
# Preprocessing
#======================================
model = Model('cantilever')

model.add_node(id='A', x=0, y=0, z=0)
model.add_node(id='B', x=0, y=1, z=0)
model.add_node(id='C', x=0, y=2, z=0)
model.add_node(id='D', x=0, y=3, z=0)
model.add_node(id='E', x=1, y=3, z=0)
model.add_node(id='F', x=1, y=2, z=0)
model.add_node(id='G', x=1, y=1, z=0)
model.add_node(id='H', x=1, y=0, z=0)

model.add_truss_element(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
model.add_truss_element(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)
model.add_truss_element(id=3, node_a='C', node_b='D', youngs_modulus=1, area=1)
model.add_truss_element(id=4, node_a='E', node_b='F', youngs_modulus=1, area=1)
model.add_truss_element(id=5, node_a='F', node_b='G', youngs_modulus=1, area=1)
model.add_truss_element(id=6, node_a='G', node_b='H', youngs_modulus=1, area=1)

model.add_truss_element(id=7, node_a='D', node_b='E', youngs_modulus=1, area=1)
model.add_truss_element(id=8, node_a='C', node_b='F', youngs_modulus=1, area=1)
model.add_truss_element(id=9, node_a='B', node_b='G', youngs_modulus=1, area=1)

model.add_cable_element(id=10, node_a='A', node_b='G', youngs_modulus=1, area=1)
model.add_cable_element(id=11, node_a='B', node_b='H', youngs_modulus=1, area=1)
model.add_cable_element(id=12, node_a='B', node_b='F', youngs_modulus=1, area=1)
model.add_cable_element(id=13, node_a='C', node_b='G', youngs_modulus=1, area=1)
model.add_cable_element(id=14, node_a='C', node_b='E', youngs_modulus=1, area=1)
model.add_cable_element(id=15, node_a='D', node_b='F', youngs_modulus=1, area=1)

model.add_single_load(id='load 1', node_id='D', fu=1.0)

model.add_dirichlet_condition(node_id='A', dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id='B', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='C', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='D', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='E', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='F', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='G', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='H', dof_types='uvw', value=0)

#======================================
# Analysis
#======================================

# define a load curve with the lambda values that should be used
load_curve = np.linspace(0.01, 0.1, 20)
for lam in load_curve:

    # create a new model for each solution step
    model = model.get_duplicate()

    predictor_method = LoadIncrementPredictor()

    path_following_method = LoadControl(lam)
    
    model.perform_non_linear_solution_step(predictor_method=predictor_method,
                                        path_following_method=path_following_method)

#======================================
# Postprocessing
#======================================

# plot the load displacement curve
plot = Plot2D()
plot.add_load_displacement_curve(model, dof=('D', 'u'))
plot.show()

# animated plot
animation = Animation3D()
animation.show(model)
