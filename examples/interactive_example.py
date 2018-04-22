# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
from nfem import *

model = Model('Initial Model')

model.add_node(id='A', x=0, y=0, z=0)
model.add_node(id='B', x=5, y=2, z=0)
model.add_node(id='C', x=10, y=0, z=0)

model.add_truss_element(id=1, node_a='A', node_b='B', youngs_modulus=10, area=2)
model.add_truss_element(id=2, node_a='B', node_b='C', youngs_modulus=10, area=2)

model.add_single_load(id='F', node_id='B', fv=-1)

model.add_dirichlet_condition(node_id='A', dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id='B', dof_types='w', value=0)
model.add_dirichlet_condition(node_id='C', dof_types='uvw', value=0)

interact(model=model, dof=('B', 'v'))
