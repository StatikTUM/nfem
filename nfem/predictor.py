import numpy as np
from .assembler import Assembler

class Predictor():

    def Predict(self, model):
        """Returns a normalized predictor"""
        raise NotImplementedError

class LoadIncrementPredictor(Predictor):

    def __init__(self, value=1.0):
        self.value = value

    def Predict(self, model):
        model.lam += self.value
        return

class DisplacementIncrementPredictor(Predictor):

    def __init__(self, dof, value=1.0):
        self.dof = dof
        self.value = value

    def Predict(self, model):
        dof_value = model.GetDofState(self.dof)
        model.SetDofState(self.dof, dof_value + self.value)
        return

class LastIncrementPredictor(Predictor): 
 
    def Predict(self, model):  
        previous_model = model.previous_model
        second_previous_model = previous_model.previous_model

        if second_previous_model == None:
            raise RuntimeError('LastIncrementPredictor can only be used after the first step.')
  
        for node in model.nodes.values(): 
            previous_node = previous_model.nodes[node.id]
            second_previous_node = second_previous_model.nodes[node.id]
            
            delta = previous_node.u - second_previous_node.u 
            node.u = previous_node.u + delta 
            
            delta = previous_node.v - second_previous_node.v 
            node.v = previous_node.v + delta 
            
            delta = previous_node.w - second_previous_node.w 
            node.w = previous_node.w + delta 
    
        delta = previous_model.lam - second_previous_model.lam 
        model.lam = previous_model.lam + delta 

# class TangentVectorPredictor(Predictor):

#     def Predict(self, model):        
#         assembler = Assembler(self)
#         dof_count = assembler.dof_count

#         u = np.zeros(dof_count)

#         for dof, value in self.dirichlet_conditions.items():
#             index = assembler.IndexOfDof(dof)
#             u[index] = value

#         k = np.zeros((dof_count, dof_count))
#         f = np.zeros(dof_count)

#         lam = 1
#         assembler.Calculate(u, lam, k, k, k, f)

#         free_count = assembler.free_dof_count

#         a = k[:free_count, :free_count]
#         b = f[:free_count] - k[:free_count, free_count:] @ u[free_count:]

#         u[:free_count] = la.solve(a, b)

#         predictor = np.array(dof_count+1)
#         predictor[:dof_count] = u
#         predictor[-1] = 1



