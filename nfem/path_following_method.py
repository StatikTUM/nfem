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
        previous_model = model.previous_model
        for (node, previous_node) in zip(model.nodes.values(), previous_model.nodes.values()):
            delta_x = node.x - previous_node.x
            delta_x *= factor
            node.x = previous_node.x + delta_x

            delta_y = node.y - previous_node.y
            delta_y *= factor
            node.y = previous_node.y + delta_y

            delta_z = node.z - previous_node.z
            delta_z *= factor
            node.z = previous_node.z + delta_z
        
        delta_lambda = model.lam - previous_model.lam
        delta_lambda *= factor
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
        initial_model = model.GetInitialModel()
        previous_model = model.previous_model
        node_id, dof_type = self.dof
        node = model.nodes[node_id]
        initial_node = initial_model.nodes[node_id]
        previous_node = previous_model.nodes[node_id]
        if dof_type == "u":
            displacement = node.x - initial_node.x
            prev_displacement = previous_node.x - initial_node.x
        elif dof_type == "v":
            displacement = node.y - initial_node.y
            prev_displacement = previous_node.x - initial_node.x
        elif dof_type == "w":
            displacement = node.z - initial_node.z
            prev_displacement = previous_node.x - initial_node.x
        
        desired_delta = self.displacement_hat - prev_displacement
        current_delta = displacement - prev_displacement
        factor = desired_delta/current_delta
        self.ScaleDeltaUandLambda(model, factor)
        return

    def CalculateConstraint(self, model):
        node_id, dof_type = self.dof
        initial_node = model.GetInitialModel().nodes[node_id]
        node = model.nodes[node_id]
        if dof_type == "u":
            displacement = node.x - initial_node.x
        elif dof_type == "v":
            displacement = node.y - initial_node.y
        elif dof_type == "w":
            displacement = node.z - initial_node.z
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
            node_id, dof_type = dof
            node = model.nodes[node_id]
            previous_node = previous_model.nodes[node_id]
            if dof_type == 'u':
                dc[i] = 2*node.x - 2*previous_node.x
            elif dof_type == 'v':
                dc[i] = 2*node.y - 2*previous_node.y
            elif dof_type == 'w':
                dc[i] = 2*node.z - 2*previous_node.z
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
