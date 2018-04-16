import numpy as np
import math
from .assembler import Assembler

class PathFollowingMethod():

    def ScalePredictor(self, model):
        """ FIXME """
        raise NotImplementedError

    def CalculateConstraint(self, model):
        """ FIXME """
        #returns c
        raise NotImplementedError

    def CalculateDerivatives(self, model):
        """ FIXME """
        #returns dc_du, dc_dLambda
        raise NotImplementedError

    def ScaleDeltaUandLambda(self, model, factor):
        assembler = Assembler(model)
        previous_model = model.previous_model

        for dof in assembler.dofs:
            current_value = model.GetDofState(dof)
            previous_value = previous_model.GetDofState(dof)

            delta = factor * (current_value - previous_value)

            model.SetDofState(dof, previous_value + delta)

        delta_lambda = factor * (model.lam - previous_model.lam)
        model.lam = previous_model.lam + delta_lambda


class LoadControl(PathFollowingMethod):
    def __init__(self, lam_hat):
        super(LoadControl, self).__init__()
        self.lam_hat = lam_hat

    def ScalePredictor(self, model):
        previous_model = model.previous_model
        desired_delta_lam = self.lam_hat - previous_model.lam
        current_delta_lam = model.lam - previous_model.lam
        factor = desired_delta_lam/current_delta_lam
        self.ScaleDeltaUandLambda(model, factor)
        return
        
    def CalculateConstraint(self, model):
        c = model.lam - self.lam_hat
        return c

    def CalculateDerivatives(self, model, dc):
        dc.fill(0.0)
        dc[-1] = 1.0
        return

class DisplacementControl(PathFollowingMethod):
    def __init__(self, node_id, dof_type, displacement_hat):
        super(DisplacementControl, self).__init__()
        self.displacement_hat = displacement_hat
        self.dof = (node_id, dof_type)

    def ScalePredictor(self, model):
        displacement = model.GetDofState(self.dof)

        previous_model = model.previous_model
        prev_displacement = previous_model.GetDofState(self.dof)
                
        desired_delta = self.displacement_hat - prev_displacement
        current_delta = displacement - prev_displacement
        factor = desired_delta/current_delta
        self.ScaleDeltaUandLambda(model, factor)
        return

    def CalculateConstraint(self, model):
        displacement = model.GetDofState(self.dof)
        c =  displacement - self.displacement_hat
        return c

    def CalculateDerivatives(self, model, dc):
        dc.fill(0.0)
        assembler = Assembler(model)
        index = assembler.IndexOfDof(self.dof)
        dc[index] = 1.0
        return

class ArcLengthControl(PathFollowingMethod):
    def __init__(self, l_hat):
        super(ArcLengthControl, self).__init__()
        self.squared_l_hat = l_hat*l_hat

    def ScalePredictor(self, model):
        squared_l = self.__CalculateSquaredPredictorLength(model)
        factor = self.squared_l_hat/squared_l
        factor = math.sqrt(factor)
        self.ScaleDeltaUandLambda(model, factor)
        return

    def CalculateConstraint(self, model):
        squared_l = self.__CalculateSquaredPredictorLength(model)
        c = squared_l - self.squared_l_hat
        return c

    def CalculateDerivatives(self, model, dc): 
        dc.fill(0.0)  
        assembler = Assembler(model)
        free_count = assembler.free_dof_count
        previous_model = model.previous_model 
        for i in range(free_count):
            dof = assembler.dofs[i]
            dc[i] = 2*model.GetDofState(dof) - 2*previous_model.GetDofState(dof)
        dc[-1] = 2*model.lam - 2*model.previous_model.lam
        return

    def __CalculateSquaredPredictorLength(self, model):
        previous_model = model.previous_model       
        squared_l = 0.0 
        for (node, previous_node) in zip(model.nodes.values(), previous_model.nodes.values()):
            dx, dy, dz = node.GetActualLocation() - previous_node.GetActualLocation()
            squared_l += dx*dx + dy*dy + dz*dz
        delta_lam = model.lam - previous_model.lam
        squared_l += delta_lam*delta_lam
        return squared_l
