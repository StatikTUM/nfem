from nfem import model
from nfem import history

model = model.Model("two_bar_truss")

# nodes
model.add_node(id='A', x= 0, y=0, z=0)
model.add_node(id='B', x= 5, y=5, z=0)
model.add_node(id='C', x=10, y=0, z=0)

# elements
model.add_truss_element(id=1, node_a='A', node_b='B')
model.add_truss_element(id=2, node_a='B', node_b='C')

# dirichlet boundary conditions
#model.add_dirichlet(id='A', node='A', type='xyz')
#model.add_dirichlet(id='B', node='B', type='z')
#model.add_dirichlet(id='C', node='C', type='xyz')

# neumann boundary conditions
#model.add_load(id='F', node='B', fy=-1)

# init history
history = history.History(model)

# simulate 10 steps
n_steps = 10
for step in range(1, n_steps):
    model.nodes['B'].y += -0.5
    history.AddModel(step, model)

# how to get e.g. the nodes of a certain step from history through nodes list
for step in range(1, n_steps):
    step_model = history.GetModel(step)
    value = step_model.nodes['B'].y
    print("Get step ", step, "value ", value)

# how to get e.g. the nodes of a certain step from history through elements list
for step in range(1, n_steps):
    step_model = history.GetModel(step)
    value = step_model.elements[1].node_b.y
    print("Get step ", step, "value ", value)

# initial value
initial_model = history.GetModel(0)
value = initial_model.elements[1].node_b.y
print("Initial value ", value)