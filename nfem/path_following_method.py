import numpy as np

from .assembler import Assembler

class PathFollowingMethod(object):
    def __init__(self, tolerance=1e-5, max_iterations=100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def scale_predictor(self, model):
        raise NotImplementedError

    def calculate_constraint(self, model):
        #returns c
        raise NotImplementedError

    def calculate_derivatives(self, model, dc):
        #returns dc_du, dc_dLambda
        raise NotImplementedError

    def scale_delta_uand_lambda(self, model, factor):
        assembler = Assembler(model)
        previous_model = model.previous_model

        for dof in assembler.dofs:
            current_value = model.get_dof_state(dof)
            previous_value = previous_model.get_dof_state(dof)

            delta = factor * (current_value - previous_value)

            model.set_dof_state(dof, previous_value + delta)

        delta_lambda = factor * (model.lam - previous_model.lam)
        model.lam = previous_model.lam + delta_lambda


class LoadControl(PathFollowingMethod):
    def __init__(self, lam_hat, *args, **kwargs):
        super(LoadControl, self).__init__(*args, **kwargs)
        self.lam_hat = lam_hat

    def scale_predictor(self, model):
        previous_model = model.previous_model
        desired_delta_lam = self.lam_hat - previous_model.lam
        current_delta_lam = model.lam - previous_model.lam
        factor = desired_delta_lam/current_delta_lam
        self.scale_delta_uand_lambda(model, factor)

    def calculate_constraint(self, model):
        return model.lam - self.lam_hat

    def calculate_derivatives(self, model, dc):
        dc.fill(0.0)
        dc[-1] = 1.0

class DisplacementControl(PathFollowingMethod):
    def __init__(self, dof, displacement_hat, *args, **kwargs):
        super(DisplacementControl, self).__init__(*args, **kwargs)
        self.displacement_hat = displacement_hat
        self.dof = dof

    def scale_predictor(self, model):
        dof = self.dof
        displacement_hat = self.displacement_hat
        previous_model = model.previous_model

        previous_displacement = previous_model.get_dof_state(dof)
        prediction_displacement = model.get_dof_state(dof)

        desired_delta = displacement_hat - previous_displacement
        current_delta = prediction_displacement - previous_displacement

        factor = desired_delta / current_delta

        self.scale_delta_uand_lambda(model, factor)

    def calculate_constraint(self, model):
        dof = self.dof
        displacement_hat = self.displacement_hat

        displacement = model.get_dof_state(dof)

        return displacement - displacement_hat

    def calculate_derivatives(self, model, dc):
        dc.fill(0.0)
        assembler = Assembler(model)
        index = assembler.index_of_dof(self.dof)
        dc[index] = 1.0

class ArcLengthControl(PathFollowingMethod):
    def __init__(self, l_hat, *args, **kwargs):
        super(ArcLengthControl, self).__init__(*args, **kwargs)
        self.squared_l_hat = l_hat**2

    def scale_predictor(self, model):
        squared_l_hat = self.squared_l_hat
        squared_l = self._calculate_squared_predictor_length(model)

        factor = np.sqrt(squared_l_hat / squared_l)

        self.scale_delta_uand_lambda(model, factor)

    def calculate_constraint(self, model):
        squared_l_hat = self.squared_l_hat
        squared_l = self._calculate_squared_predictor_length(model)

        return squared_l - squared_l_hat

    def calculate_derivatives(self, model, dc):
        dc.fill(0.0)

        assembler = Assembler(model)
        previous_model = model.previous_model

        for index, dof in enumerate(assembler.free_dofs):
            current_value = model.get_dof_state(dof)
            previous_value = previous_model.get_dof_state(dof)

            dc[index] = 2 * (current_value - previous_value)

        dc[-1] = 2 * (model.lam - model.previous_model.lam)

    def _calculate_squared_predictor_length(self, model):
        previous_model = model.previous_model

        squared_l = 0.0

        for node, previous_node in zip(model.nodes.values(), previous_model.nodes.values()):
            dx, dy, dz = node.get_actual_location() - previous_node.get_actual_location()
            squared_l += dx**2 + dy**2 + dz**2

        delta_lam = model.lam - previous_model.lam

        squared_l += delta_lam**2

        return squared_l
