import numpy as np

class PathFollowingMethod():
    def __init__(self):
        pass

    def ScalePredictor(self, model):
        raise NotImplementedError

    def CalculateConstraint(self, model):
        #returns c
        raise NotImplementedError

    def CalculateDerivatives(self, model):
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
        #returns new_u, new_Lambda
        previous_model = model.previous_model
        delta_lambda = model.lam - previous_model.lam
        factor = self.Lambda_hat/delta_lambda
        self.ScaleDeltaUandLambda(model, factor)
        return model

    def CalculateConstraint(self, previous_u, previous_Lambda, current_u, current_Lambda):
        c = Lambda - self.Lambda_hat
        return c

    def CalculateDerivatives(self, previous_u, previous_Lambda, current_u, current_Lambda):
        dc_du = 0
        dc_dLambda = 1
        return dc_du, dc_dLambda

class ArcLengthControl(PathFollowingMethod):
    def __init__(self, l_hat):
        super(ArcLengthControl, self).__init__()
        self.l_hat = l_hat

    def Predict(self, previous_u, previous_Lambda, current_u, current_Lambda):
        #returns new_u, new_Lambda
        predicted_u = u
        predicted_Lambda = self.Lambda_hat
        return predicted_u, predicted_Lambda

    def CalculateConstraint(self, previous_u, previous_Lambda, current_u, current_Lambda):
        c = Lambda - self.Lambda_hat
        return c

    def CalculateDerivatives(self, previous_u, previous_Lambda, current_u, current_Lambda):
        dc_du = 0
        dc_dLambda = 1
        return dc_du, dc_dLambda
