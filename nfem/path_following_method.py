import numpy as np

from .assembler import Assembler

class PathFollowingMethod():

    def ScalePredictor(self, model):
        raise NotImplementedError

    def CalculateConstraint(self, model):
        #returns c
        raise NotImplementedError

    def CalculateDerivatives(self, model, dc):
        #returns dc_du, dc_dLambda
        raise NotImplementedError

    def ScaleDeltaUandLambda(self, model, factor):
        previous_model = model.previous_model
        for (node, previous_node) in zip(model.nodes.values(), previous_model.nodes.values()):
            delta_u = node.u - previous_node.u
            delta_u *= factor
            node.u = previous_node.u + delta_u

            delta_v = node.v - previous_node.v
            delta_v *= factor
            node.v = previous_node.v + delta_v

            delta_w = node.w - previous_node.w
            delta_w *= factor
            node.w = previous_node.w + delta_w

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

    def CalculateConstraint(self, model):
        return model.lam - self.lam_hat

    def CalculateDerivatives(self, model, dc):
        dc.fill(0.0)
        dc[-1] = 1.0

class DisplacementControl(PathFollowingMethod):
    def __init__(self, dof, displacement_hat):
        super(DisplacementControl, self).__init__()
        self.displacement_hat = displacement_hat
        self.dof = dof

    def ScalePredictor(self, model):
        dof = self.dof
        displacement_hat = self.displacement_hat
        previous_model = model.previous_model

        previous_displacement = previous_model.GetDofState(dof)
        prediction_displacement = model.GetDofState(dof)

        desired_delta = displacement_hat - previous_displacement
        current_delta = prediction_displacement - previous_displacement

        factor = desired_delta / current_delta

        self.ScaleDeltaUandLambda(model, factor)

    def CalculateConstraint(self, model):
        dof = self.dof
        displacement_hat = self.displacement_hat

        displacement = model.GetDofState(dof)

        return displacement - displacement_hat

    def CalculateDerivatives(self, model, dc):
        dc.fill(0.0)
        assembler = Assembler(model)
        index = assembler.IndexOfDof(self.dof)
        dc[index] = 1.0

class ArcLengthControl(PathFollowingMethod):
    def __init__(self, l_hat):
        super(ArcLengthControl, self).__init__()
        self.squared_l_hat = l_hat**2

    def ScalePredictor(self, model):
        squared_l_hat = self.squared_l_hat
        squared_l = self._CalculateSquaredPredictorLength(model)

        factor = np.sqrt(squared_l_hat / squared_l)

        self.ScaleDeltaUandLambda(model, factor)

    def CalculateConstraint(self, model):
        squared_l_hat = self.squared_l_hat
        squared_l = self._CalculateSquaredPredictorLength(model)

        return squared_l - squared_l_hat

    def CalculateDerivatives(self, model, dc):
        dc.fill(0.0)

        assembler = Assembler(model)
        free_count = assembler.free_dof_count
        previous_model = model.previous_model

        for index, dof in enumerate(assembler.dofs[:free_count]):
            current_value = model.GetDofState(dof)
            previous_value = previous_model.GetDofState(dof)

            dc[index] = 2 * (current_value - previous_value)

        dc[-1] = 2*model.lam - 2*model.previous_model.lam

    def _CalculateSquaredPredictorLength(self, model):
        previous_model = model.previous_model

        squared_l = 0.0

        for node, previous_node in zip(model.nodes.values(), previous_model.nodes.values()):
            dx, dy, dz = node.GetActualLocation() - previous_node.GetActualLocation()
            squared_l += dx**2 + dy**2 + dz**2

        delta_lam = model.lam - previous_model.lam

        squared_l += delta_lam**2

        return squared_l
