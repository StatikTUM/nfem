from nfem import model
from nfem import history
from nfem import plotting_utility

model = model.Model("two_bar_truss")

# nodes
model.AddNode(id='A', x= 0, y=0, z=0)
model.AddNode(id='B', x= 5, y=5, z=0)
model.AddNode(id='C', x=10, y=0, z=0)

# elements
model.AddTrussElement(id=1, node_a='A', node_b='B', youngs_modulus=1000, area=1)
model.AddTrussElement(id=2, node_a='B', node_b='C', youngs_modulus=1000, area=1)

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
    model.nodes['B'].x += -0.3
    model.nodes['B'].z += 0.1
    history.AddModel(step, model)

plotting_utility.plot_cont_animated(history,speed=100)



