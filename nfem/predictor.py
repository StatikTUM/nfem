import numpy as np
from .assembler import Assembler

class Predictor():

    def __init__(self):

    def GetPredictor(self, model):
        """Returns a normalized predictor"""
        raise NotImplementedError

class LoadIncrementPredictor(Predictor):

    def GetPredictor(self, model):
        assembler = Assembler(model)
        dof_count = assembler.dof_count
        predictor = np.zeros(dof_count+1)
        predictor[-1] = 1
        return

class LoadIncrementPredictor(Predictor):

    def GetPredictedModel(self, model):
        model = model.Duplicate()
        model.name = 'Predictor'
        if model.lam == None:
            model.lam = 0.0
        model.lam += 1.0
        return model
        
class DisplacementIncrementPredictor(Predictor):

    def __init__(self, node_id, dof_type):
        if len(dof_type) != 1:
            raise RuntimeError('Only single dof can be incremented by this predictor')
        self.dof = (node_id, dof_type)

    def GetPredictor(self, model):
        assembler = Assembler(model)
        dof_count = assembler.dof_count
        predictor = np.zeros(dof_count+1)
        index = assembler.IndexOfDof(self.dof)
        predictor[index] = 1
        return

class LastDeltaPredictor(Predictor):

    def __init__(self, history):
        self.history = history

    def GetPredictor(self, model):
        assembler = Assembler(model)
        dof_count = assembler.dof_count
        predictor = np.zeros(dof_count+1)

        previous_model = self.history.GetModel(-2)

        # TODO loop dofs from assembler
        # Node.Get

        for node in model.nodes:
            previous_node = previous_model.nodes[node.id]

            index = assembler.IndexOfDof((node.id, 'u'))
            delta = node.x - previous_node.x
            predictor[index] = delta
            
            index = assembler.IndexOfDof((node.id, 'v'))
            delta = node.y - previous_node.y
            predictor[index] = delta
            
            index = assembler.IndexOfDof((node.id, 'w'))
            delta = node.x - previous_node.x
            predictor[index] = delta

        predictor[-1] = model.lam - previous_model.lam

class TangentVectorPredictor(Predictor):

    def GetPredictor(self, model):        
        assembler = Assembler(self)
        dof_count = assembler.dof_count

        u = np.zeros(dof_count)

        for dof, value in self.dirichlet_conditions.items():
            index = assembler.IndexOfDof(dof)
            u[index] = value

        k = np.zeros((dof_count, dof_count))
        f = np.zeros(dof_count)

        lam = 1
        assembler.Calculate(u, lam, k, k, k, f)

        free_count = assembler.free_dof_count

        a = k[:free_count, :free_count]
        b = f[:free_count] - k[:free_count, free_count:] @ u[free_count:]

        u[:free_count] = la.solve(a, b)

        predictor = np.array(dof_count+1)
        predictor[:dof_count] = u
        predictor[-1] = 1


    
# TODO different methods to predict the solution:
# 1. load increment, u remains same
# 2. displacement imcrement, lambda remains same
# 3. [delta_u, delta_lambda] -> requires a previous solution
# 4. use tangent stiffness at current equilibrium point
# 5. some manual values

LoadIncrementPredictor().GetPredictor(None)