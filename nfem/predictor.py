"""This module contains prediction strategies for solving nonlinear systems.

Author: Armin Geiser
"""

import numpy as np
from .assembler import Assembler

class Predictor():

    def predict(self, model):        
        raise NotImplementedError

class LoadIncrementPredictor(Predictor):
    """The LoadIncrementPredictor predicts a solution by only incrementing the 
        load by a certain delta lambda

    Attributes
    ----------
    value : float
        Value that is used to increment the load factor lambda at the model. 
        1.0 by default.
    """

    def __init__(self, value=1.0):
        """Create a new LoadIncrementPredictor

        Parameters
        ----------
        value : float
            Value that is used to increment the load factor lambda at the model. 
            1.0 by default.
        """
        self.value = value

    def predict(self, model):
        """Predicts the solution by incrementing lambda

        Parameters
        ----------
        model : Model
            Model to predict.
        """
        model.lam += self.value
        return

class DisplacementIncrementPredictor(Predictor):
    """The DisplacementIncrementPredictor predicts a solution by only incrementing 
        the displacement of a single dof.
    
    Attributes
    ----------
    dof : object
        Dof that is incremented.
    value : float
        Value that is used to increment the dof at the model. 1.0 by default.
    """

    def __init__(self, dof, value=1.0):
        """Create a new DisplacementIncrementPredictor

        Parameters
        ----------
        dof : object
            Dof that is incremented.
        value : float
            Value that is used to increment the dof at the model. 1.0 by default.
        """
        self.dof = dof
        self.value = value

    def predict(self, model):
        """Predicts the solution by incrementing the dof

        Parameters
        ----------
        model : Model
            Model to predict.
        """
        dof_value = model.get_dof_state(self.dof)
        model.set_dof_state(self.dof, dof_value + self.value)
        return

class LastIncrementPredictor(Predictor): 
    """The LastIncrementPredictor predicts a solution by using the last increment 
        of lambda and all dofs of the models history. It can only be used after 
        the first step.
    """
 
    def predict(self, model):  
        """Predicts the solution by incrementing lambda and all dofs with the 
            last increment

        Parameters
        ----------
        model : Model
            Model to predict.

        Raises
        ------
        RuntimeError
            If the model has not already one calculated step.
        """
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

#     def predict(self, model):        
#         assembler = Assembler(self)
#         dof_count = assembler.dof_count

#         u = np.zeros(dof_count)

#         for dof, value in self.dirichlet_conditions.items():
#             index = assembler.index_of_dof(dof)
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



