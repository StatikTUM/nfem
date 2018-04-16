# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
from nfem import *

model = Model('Initial Model')

model.AddNode(id='A', x=0, y=0, z=0)
model.AddNode(id='B', x=5, y=2, z=0)
model.AddNode(id='C', x=10, y=0, z=0)

model.AddTrussElement(id=1, node_a='A', node_b='B', youngs_modulus=10, area=2)
model.AddTrussElement(id=2, node_a='B', node_b='C', youngs_modulus=10, area=2)

model.AddSingleLoad(id='F', node_id='B', fv=-1)

model.AddDirichletCondition(node_id='A', dof_types='uvw', value=0)
model.AddDirichletCondition(node_id='B', dof_types='w', value=0)
model.AddDirichletCondition(node_id='C', dof_types='uvw', value=0)

Interact(model=model, dof=('B', 'v'))
