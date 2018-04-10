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
    def __init__(self, node_1,node_2):
        self.node_1 = node_1
        self.node_2 = node_2


class Model(object):
    def __init__(self):
        self.elements = []

node_1 = Node(0, 0)
node_2 = Node(1, 1)
node_3 = Node(2, 0)
element_1 = Element(node_1, node_2)
element_2 = Element(node_2, node_3)

model = Model()
model.elements.append(element_1)
model.elements.append(element_2)

###############################
step = 0
history = History(model)
step += 1
model.elements[0].node_2.y = 0.75
history.AddModel(step, model)
step += 1
model.elements[0].node_2.y = 0.5
history.AddModel(step, model)
step += 1
model.elements[0].node_2.y = 0.25
history.AddModel(step, model)

print("Initial value: ", history.GetModel(0).elements[0].node_2.y)
print("Step 1 value: ", history.GetModel(1).elements[0].node_2.y)
print("Step 2 value: ", history.GetModel(2).elements[0].node_2.y)
print("Step 3 value: ", history.GetModel(3).elements[0].node_2.y)
print("Current model value: ", model.elements[0].node_2.y)
