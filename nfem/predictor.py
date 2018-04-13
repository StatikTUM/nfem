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

    def __init__(self, node_id=2, dof_type='v', value=1.0):
        if len(dof_type) != 1:
            raise RuntimeError('Only single dof can be incremented by this predictor')
        self.dof = (node_id, dof_type)
        self.value = value

    def Predict(self, model):
        node_id, dof_type = self.dof
        node = model.nodes[node_id]
        if dof_type == "u":
            node.x += self.value
        elif dof_type == "v":
            node.y += self.value
        elif dof_type == "w":
            node.z += self.value
        return


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



