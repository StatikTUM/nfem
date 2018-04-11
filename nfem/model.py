class Node(object):
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.reference_x = x
        self.reference_y = y
        self.reference_z = z

class Truss(object):
    def __init__(self, id, node_a, node_b):
        self.id = id
        self.node_a = node_a
        self.node_b = node_b

class Model(object):
    def __init__(self, name):
        self.name = name
        self.nodes = dict()
        self.elements = dict()
        self.dirichlet_conditions = dict()
        self.neumann_conditions = dict()

    def add_node(self, id, x, y, z):
        if id in self.nodes:
            raise RuntimeError("Node with given id already esists")
        else:
            self.nodes[id] = Node(id, x, y, z)

    def add_truss_element(self, id, node_a, node_b):
        if id in self.elements:
            raise RuntimeError("Element with given id already esists")
        else:
            self.elements[id] = Truss(id, self.nodes[node_a], self.nodes[node_b])