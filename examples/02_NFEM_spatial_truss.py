"""
from Tr_02_SpatialTruss.xlsx
"""

from nfem import *

model = Model('Spatial truss')

model.add_node(id=1, x=0.000000000, y=10.000000000, z=0.000000000)
model.add_node(id=2, x=9.510565163, y=3.090169944, z=0.000000000)
model.add_node(id=3, x=5.877852523, y=-8.090169944, z=0.000000000)
model.add_node(id=4, x=-5.877852523, y=-8.090169944, z=0.000000000)
model.add_node(id=5, x=-9.510565163, y=3.090169944, z=0.000000000)
model.add_node(id=6, x=3.526711514, y=4.854101966, z=1.98057153, fz=-1)
model.add_node(id=7, x=5.706339098, y=-1.854101966, z=1.98057153, fz=-1)
model.add_node(id=8, x=7.35089E-16, y=-6.000000000, z=1.98057153, fz=-1)
model.add_node(id=9, x=-5.706339098, y=-1.854101966, z=1.98057153, fz=-1)
model.add_node(id=10, x=-3.526711514, y=4.854101966, z=1.98057153, fz=-1)
model.add_node(id=11, x=0.000000000, y=0.000000000, z=3.000000000, fz=-1)

model.add_truss_element(id=1  , node_a= 1   , node_b=  6, youngs_modulus=1, area=1)
model.add_truss_element(id=2  , node_a= 6   , node_b=  2, youngs_modulus=1, area=1)
model.add_truss_element(id=3  , node_a= 2   , node_b=  7, youngs_modulus=1, area=1)
model.add_truss_element(id=4  , node_a= 7   , node_b=  3, youngs_modulus=1, area=1)
model.add_truss_element(id=5  , node_a= 3   , node_b=  8, youngs_modulus=1, area=1)
model.add_truss_element(id=6  , node_a= 8   , node_b=  4, youngs_modulus=1, area=1)
model.add_truss_element(id=7  , node_a= 4   , node_b=  9, youngs_modulus=1, area=1)
model.add_truss_element(id=8  , node_a= 9   , node_b=  5, youngs_modulus=1, area=1)
model.add_truss_element(id=9  , node_a= 5   , node_b= 10, youngs_modulus=1, area=1)
model.add_truss_element(id=10 , node_a=  10 , node_b=  1, youngs_modulus=1, area=1)
model.add_truss_element(id=11 , node_a=  6  , node_b=  7, youngs_modulus=1, area=1)
model.add_truss_element(id=12 , node_a=  7  , node_b=  8, youngs_modulus=1, area=1)
model.add_truss_element(id=13 , node_a=  8  , node_b=  9, youngs_modulus=1, area=1)
model.add_truss_element(id=14 , node_a=  9  , node_b= 10, youngs_modulus=1, area=1)
model.add_truss_element(id=15 , node_a=  10 , node_b=  6, youngs_modulus=1, area=1)
model.add_truss_element(id=16 , node_a=  6  , node_b= 11, youngs_modulus=1, area=1)
model.add_truss_element(id=17 , node_a=  7  , node_b= 11, youngs_modulus=1, area=1)
model.add_truss_element(id=18 , node_a=  8  , node_b= 11, youngs_modulus=1, area=1)
model.add_truss_element(id=19 , node_a=  9  , node_b= 11, youngs_modulus=1, area=1)
model.add_truss_element(id=20 , node_a=  10 , node_b= 11, youngs_modulus=1, area=1)

model.add_dirichlet_condition(node_id=1, dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id=2, dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id=3, dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id=4, dof_types='uvw', value=0)
model.add_dirichlet_condition(node_id=5, dof_types='uvw', value=0)

model = interact(model=model, dof=(11, 'w'))
