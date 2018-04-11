from nfem import model
from nfem import history
from nfem import plotting_utility as plt 

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
n_steps = 15
for step in range(1, n_steps):
    model.nodes['B'].y += -0.5
    model.nodes['B'].x += -0.3
    model.nodes['B'].z += 0.1
    history.AddModel(step, model)

plt.plot_cont_animated(history,speed=200)



