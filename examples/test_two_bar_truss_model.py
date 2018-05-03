import sys
sys.path.append('..') 

# import necessary modules
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
# Test
#======================================
assert(len(model._nodes) == 3)
assert(len(model.elements) == 3)
assert(len(model.dirichlet_conditions) == 7)

#======================================
# Export model for further use
#======================================
def get_model():
    return model