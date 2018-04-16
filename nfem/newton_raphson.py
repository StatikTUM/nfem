"""FIXME"""

import numpy.linalg as la

class NewtonRaphson(object):
    """FIXME"""

    def __init__(self, max_iterations=100, tolerance=1e-9):
        """FIXME"""
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def Solve(self, calculate_system, x_initial):
        """FIXME"""
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

        raise RuntimeError('Newthon-Raphson did not converge after {} steps. Residual norm: {}' \
                           .format(self.max_iterations, residual_norm))
