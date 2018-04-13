import numpy as np

class NewtonRaphson(object):

    def __init__(self, max_iterations=10, tolerance=1e-9):
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
            print('Residual norm:', residual_norm)
            if residual_norm < self.tolerance:
                print('Newthon-Rapshon converged in step {}.'.format(i))
                return x

            # compute delta_x
            delta_x = np.linalg.solve(LHS, RHS)

            # update x
            x -= delta_x

        raise RuntimeError('Newthon-Rapshon did not converge after {} steps. Residual norm: {}'.format(self.max_iterations, residual_norm))
        return x

# def CalculateSystem(x):
#     print("X:",x)
#     RHS = np.zeros(1)
#     RHS[0] = np.power(x[0],7) - 1000.
# 
#     LHS = np.zeros((1,1))
#     LHS[0,0] = 7*np.power(x[0],6)
#     return LHS, RHS
# 
# x_start = np.zeros(1)
# x_start[0] = 3.0
# NewtonRaphson().Solve(CalculateSystem, x_start)
