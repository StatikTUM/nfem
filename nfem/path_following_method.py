import numpy as np
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
    def __init__(self, Lambda_hat):
        super(LoadControl, self).__init__()
        self.Lambda_hat = Lambda_hat

    def ScalePredictor(self, model):
        previous_model = model.previous_model
        delta_lambda = model.lam - previous_model.lam
        factor = self.Lambda_hat/delta_lambda
        self.ScaleDeltaUandLambda(model, factor)
        return model

    def CalculateConstraint(self, model):
        c = model.lam - self.Lambda_hat
        return c

    def CalculateDerivatives(self, model, dc):
        dc.fill(0.0)
        dc[-1] = 1.0
        return dc

class DisplacementControl(PathFollowingMethod):
    def __init__(self, u_hat):
        super(DisplacementControl, self).__init__()
        self.u_hat = u_hat
        self.dof = (2,'v')

    def ScalePredictor(self, model):
        #returns new_u, new_Lambda
        previous_model = model.previous_model
        node = model.nodes[self.dof[0]]
        previous_node = previous_model.nodes[self.dof[0]]
        delta_u = node.y - previous_node.y # TODO get according to dof
        factor = self.u_hat/delta_u
        self.ScaleDeltaUandLambda(model, factor)
        return model

    def CalculateConstraint(self, model):
        node = model.nodes[self.dof[0]]
        c = node.y - self.u_hat  # TODO get according to dof
        return c

    def CalculateDerivatives(self, model, dc):
        dc.fill(0.0)
        assembler = Assembler(model)
        index = assembler.IndexOfDof(self.dof)
        dc[index] = 1.0
        return dc

class ArcLengthControl(PathFollowingMethod):
    def __init__(self, l_hat):
        super(ArcLengthControl, self).__init__()
        self.squared_l_hat = l_hat*l_hat

    def ScalePredictor(self, model):
        squared_l = self.__CalculateSquaredPredictorLength(model)
        factor = self.squared_l_hat/squared_l
        self.ScaleDeltaUandLambda(model, factor)
        return model

    def CalculateConstraint(self, model):
        squared_l = self.__CalculateSquaredPredictorLength(model)
        c = squared_l - self.squared_l_hat
        return c

    def CalculateDerivatives(self, model, dc):   
        assembler = Assembler(model)
        free_count = assembler.free_dof_count
        for i in range(free_count):
            dof = assembler.dofs[i]
            node = model.nodes[dof[0]]
            if dof[1] == 'u':
                dc[i] = node.x - node.reference_x
            elif dof[1] == 'v':
                dc[i] = node.y - node.reference_y
            elif dof[1] == 'w':
                dc[i] = node.z - node.reference_z
        dc[-1] = model.lam
        return dc

    def __CalculateSquaredPredictorLength(self, model):
        previous_model = model.previous_model       
        squared_l = 0.0 
        for (node, previous_node) in zip(model.nodes.values(), previous_model.nodes.values()):
            dx, dy, dz = node.GetActualLocation() - previous_node.GetActualLocation()
            squared_l += dx*dx + dy*dy + dz*dz
        return squared_l
