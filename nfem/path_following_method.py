"""This module contains path-following strategies for solving nonlinear systems.

Author: Armin Geiser
"""

from nfem.assembler import Assembler


class LoadControl:
    """The LoadControl adds a constraint to the non linear problem that ensures
        a prescribed load factor at the equilibrium point.

    Attributes
    ----------
    lam_hat : float
        Prescribed value for the load factor
    """

    def __init__(self, model):
        """Create a new LoadControl

        Parameters
        ----------
        lam_hat : float
            Prescribed value for the load factor
        """
        super(LoadControl, self).__init__()
        self.lam_hat = model.load_factor

    def calculate_constraint(self, model):
        """Calculates the constraint

        Parameters
        ----------
        model : Model
            Model for which the constraint is calculated.

        Returns
        ----------
        constraint : float
            Value of the constraint.
        """
        return model.load_factor - self.lam_hat

    def calculate_derivatives(self, model, dc):
        """Assembles the constraint derivative [dc/du, dc/dlam] into the vector dc

        Parameters
        ----------
        model : Model
            Model for which the constraint derivative is calculated.
        dc : ndarray
            System vector to store the results. Existing values are overwritten.
        """
        dc.fill(0.0)
        dc[-1] = 1.0


class DisplacementControl:
    """The DisplacementControl adds a constraint to the non linear problem that ensures
        a prescribed displacement of the controlled dof at the equilibrium point.

    Attributes
    ----------
    dof : Object
        Controlled dof
    displacement_hat : float
        Prescribed value for the controlled dof
    """
    def __init__(self, model, dof):
        """Create a new DisplacementControl

        Parameters
        ----------
        dof : Object
            Controlled dof
        displacement_hat : float
            Prescribed value for the controlled dof
        """
        super(DisplacementControl, self).__init__()
        self.displacement_hat = model[dof].delta
        self.dof = dof

    def calculate_constraint(self, model):
        """Calculates the constraint

        Parameters
        ----------
        model : Model
            Model for which the constraint is calculated.

        Returns
        ----------
        constraint : float
            Value of the constraint.
        """
        dof = self.dof
        displacement_hat = self.displacement_hat

        displacement = model[dof].delta

        return displacement - displacement_hat

    def calculate_derivatives(self, model, dc):
        """Assembles the constraint derivative [dc/du, dc/dlam] into the vector dc

        Parameters
        ----------
        model : Model
            Model for which the constraint derivative is calculated.
        dc : ndarray
            System vector to store the results. Existing values are overwritten.
        """
        dc.fill(0.0)
        assembler = Assembler(model)
        index = assembler.index_of_dof(self.dof)
        dc[index] = 1.0


class ArcLengthControl:
    """The ArcLengthControl adds a constraint to the non linear problem that ensures
        a prescribed length of the increment between the new and the last equilibrium point.

    Attributes
    ----------
    l_hat : float
        Prescribed value for the length of the increment (l2 norm)
    """
    def __init__(self, model):
        """Create a new ArcLengthControl

        Parameters
        ----------
        l_hat : float
            Prescribed value for the length of the increment (l2 norm)
        """
        super(ArcLengthControl, self).__init__()
        self.squared_l_hat = self._calculate_squared_predictor_length(model)

    def calculate_constraint(self, model):
        """Calculates the constraint

        Parameters
        ----------
        model : Model
            Model for which the constraint is calculated.

        Returns
        ----------
        constraint : float
            Value of the constraint.
        """
        squared_l_hat = self.squared_l_hat
        squared_l = self._calculate_squared_predictor_length(model)

        return squared_l - squared_l_hat

    def calculate_derivatives(self, model, dc):
        """Assembles the constraint derivative [dc/du, dc/dlam] into the vector dc

        Parameters
        ----------
        model : Model
            Model for which the constraint derivative is calculated.
        dc : ndarray
            System vector to store the results. Existing values are overwritten.
        """
        dc.fill(0.0)

        assembler = Assembler(model)
        previous_model = model.get_previous_model()

        for index, dof in enumerate(assembler.dofs):
            current_value = model[dof].delta
            previous_value = previous_model[dof].delta

            dc[index] = 2 * (current_value - previous_value)

        dc[-1] = 2 * (model.load_factor - model.get_previous_model().load_factor)

    def _calculate_squared_predictor_length(self, model):
        previous_model = model.get_previous_model()

        squared_l = 0.0

        for node, previous_node in zip(model.nodes, previous_model.nodes):
            dx, dy, dz = node.location - previous_node.location
            squared_l += dx**2 + dy**2 + dz**2

        delta_lam = model.load_factor - previous_model.load_factor

        squared_l += delta_lam**2

        return squared_l
