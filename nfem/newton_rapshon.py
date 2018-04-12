class NewtonRaphson(object):

    def __init__(self, max_iterations=100, tolerance=1e-9):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def Solve(self, compute_LHS, compute_RHS, x):
        # loop
            # compute_LHS(x) -> gives numpy matrix (n_dof+1,n_dof+1)
            # compute_LHS(x) -> gives numpy matrix (n_dof+1,1)
            # update x       -> is numpy matrix (n_dof+1,n_dof+1)
        return x
