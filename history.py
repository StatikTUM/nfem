import time
import copy

class History(object):

    def __init__(self, model):
        self.history = []
        self.AddModel(0, model)

    def GetModel(self, step):
        return self.history[step]

    def AddModel(self, step, model):
        if len(self.history) != step:
            raise RuntimeError("History has the wrong size!")
        self.history.append(copy.deepcopy(model))

    def ResetHistoryToStep(self, step):
        self.history = self.history[0:step]
        return self.GetModel(step)


class Node(object):
    def __init__(self, x,y):
        self.x = x
        self.y = y

class Element(object):
    def __init__(self, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2


class Model(object):
    def __init__(self):
        self.elements = []

# TESTING
n_elements = 100
n_steps = 1000

# create the initial model
model = Model()
for i in range(0, n_elements):
    node_1 = Node(0, 0)
    node_2 = Node(1, 1)
    element_1 = Element(node_1, node_2)
    model.elements.append(element_1)

# init history
history = History(model)

start = time.time()
for step in range(1, n_steps):
    #print("Add step ", step)
    model.elements[0].node_2.y *= 0.99
    history.AddModel(step, model)
end = time.time()
print("Adding models", end - start)

start = time.time()
for step in range(1, n_steps):
    value = history.GetModel(step).elements[0].node_2.y
    #print("Get step ", step, "value ", value)
end = time.time()
print("Using models", end - start)

print("Initial value: ", history.GetModel(0).elements[0].node_2.y)
print("Current model value: ", model.elements[0].node_2.y)

