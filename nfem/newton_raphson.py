"""This module contains the NewtonRaphson class

Author: Armin Geiser
"""

import numpy.linalg as la

class NewtonRaphson(object):
    """NewtonRaphson solves a non linear system of equations iteratively.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance value for the residual norm
    """

    def __init__(self, max_iterations=100, tolerance=1e-9):
        """Create a new NewtonRaphson.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance value for the residual norm
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, calculate_system, x_initial):
        """Solves the nonlinear system defined by the `calculate_system` callback.
            The array with the initial solution is updated during the solve and contains 
            the solution at convergence. 

        Parameters
        ----------
        calculate_system : function callback        
            This function is called several times to evaluate the function (rhs) and the 
            functions derivatives (lhs) with a given state(x)

            It should look like this:
            def calculate_system(x):
                ...
                return lhs, rhs
        x_initial : numpy.ndarray
            Initial guess of the solution

        Raises
        ----------
        RuntimeError
            If the algorithm does not converge within `max_iterations`        
        """
        x = x_initial
        residual_norm = None

        for i in range(1, self.max_iterations + 1):

            # calculate left and right hand side
            lhs, rhs = calculate_system(x)

            # calculate residual
            residual_norm = la.norm(rhs)

            # check convergence
            if residual_norm < self.tolerance:
                print('  Newthon-Raphson converged in step {}.'.format(i))
                print('  Residual norm: {}.'.format(residual_norm))
                return x, i

            # compute delta_x
            delta_x = la.solve(lhs, rhs)

            # update x
            x -= delta_x

        raise RuntimeError('Newthon-Raphson did not converge after {} steps. Residual norm: {}'
                           .format(self.max_iterations, residual_norm))
