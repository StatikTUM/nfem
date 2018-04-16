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
