import numpy as np

class NewtonRaphson(object):

    def __init__(self, max_iterations=100, tolerance=1e-9):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def Solve(self, CalculateSystem, x_initial):
        x = x_initial
        residual_norm = None
        for i in range(1,self.max_iterations+1):

            # calculate left and right hand side
            LHS, RHS = CalculateSystem(x)

            # check residual
            residual_norm = np.linalg.norm(RHS)
            #print('Residual norm:', residual_norm)
            if residual_norm < self.tolerance:
                print(' Newthon-Raphson converged in step {}.'.format(i))
                print(' Residual norm: {}.'.format(residual_norm))
                return x, i

            # compute delta_x
            delta_x = np.linalg.solve(LHS, RHS)

            # update x
            x -= delta_x

        raise RuntimeError('Newthon-Raphson did not converge after {} steps. Residual norm: {}'.format(self.max_iterations, residual_norm))
        
