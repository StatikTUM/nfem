from nfem import Model, Assembler
import numpy as np

model = Model('Initial Model')
model.AddNode(id='A', x= 0, y=0, z=0)
model.AddNode(id='B', x= 5, y=2, z=0)
model.AddNode(id='C', x=10, y=0, z=0)
model.AddTrussElement(id=1, node_a='A', node_b='B', youngs_modulus=10, area=2)
model.AddTrussElement(id=2, node_a='B', node_b='C', youngs_modulus=10, area=2)
model.AddDirichletCondition(node='A', dof_types='uvw', value=0)
model.AddDirichletCondition(node='B', dof_types='w'  , value=0)
model.AddDirichletCondition(node='C', dof_types='uvw', value=0)
model.AddSingleLoad(id='F', node='B', fv=-1)

deformed = model.PerformLinearSolutionStep()

print('Initial:')
print(model.nodes['B'].x)
print(model.nodes['B'].y)
print(model.nodes['B'].z)

print('Deformed:')
print(deformed.nodes['B'].x)
print(deformed.nodes['B'].y)
print(deformed.nodes['B'].z)
