import numpy as np

class NewtonRaphson(object):

    def __init__(self, max_iterations=100, tolerance=1e-9):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def Solve(self, UpdateModel, compute_LHS, compute_RHS, x_initial):
        x = x_initial
        residual_norm = None
        for i in range(1,self.max_iterations):
            # check residual
            RHS = compute_RHS(x)
            residual_norm = np.linalg(RHS)
            if RHS < self.tolerance:
                print('Newthon-Rapshon converged in step {}.'.format(i))
                print('Residual norm:', residual_norm)
                return x
            # compute delta_x
            LHS = compute_LHS(x)
            delta_x = np.linalg.solve(LHS, RHS)
            # update x
            x = x + delta_x
            # call update model callback
            UpdateModel(x)
        raise RuntimeError('Newthon-Rapshon did not converge after {} steps. Residual norm: {}'.format(self.max_iterations, residual_norm))
        return x