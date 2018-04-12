from abc import ABC, abstractmethod
import numpy as np

class PathFollowingMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def Predict(self, previous_u, previous_Lambda, current_u, current_Lambda):
        # TODO predictor direction as input
        # TODO scale this direction according to the method
        # TODO renaming to scale predictor? (does not have to be called)
        #returns new_u, new_Lambda
        return NotImplemented

    @abstractmethod
    def CalculateConstraint(self, previous_u, previous_Lambda, current_u, current_Lambda):
        #returns c
        return NotImplemented

    @abstractmethod
    def CalculateDerivatives(self, previous_u, previous_Lambda, current_u, current_Lambda):
        #returns dc_du, dc_dLambda
        return NotImplemented

class LoadControl(PathFollowingMethod):
    def __init__(self, Lambda_hat):
        super(LoadControl, self).__init__()
        self.Lambda_hat = Lambda_hat

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


u = np.array([0,0,0,0])
Lambda = 0

previous_u = u
previous_Lambda = Lambda


lc = LoadControl([0,0,0,0], 0.1)
print(lc.Predict(0,1))