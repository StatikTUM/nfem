
# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
import numpy as np
import numpy.linalg as la

from nfem import NewtonRaphson

EA = 1.0
a = 1.0 
L = np.sqrt(2.0)
F = -1.0


dof_count = 1
free_count = 1
# initialize working matrices and functions for newton raphson
k = np.zeros((dof_count,dof_count))
f = np.zeros(dof_count)

LHS = np.zeros((free_count+1, free_count+1))
RHS = np.zeros(free_count+1)

lam_hat = 0.1
x = np.zeros(free_count+1)
x[0] = -0.080071057
x[-1] = lam_hat

def CalculateSystem(x):
    print("x:", x)
    u = x[0]
    lam = x[1]

    k[0,0] = EA*(a/L)**3*(2/a+6*u/a/a+3*u*u/a/a/a)
    f[0] = F

    # assemble left and right hand side for newton raphson
    LHS[:free_count, :free_count] = k[:free_count, :free_count]
    
    #variante 1 is wrong!
    #U = np.zeros(1)
    #U[0] = u
    #RHS[:free_count] = k[:free_count, free_count:] @ U[free_count:] - lam*f[:free_count]

    #variante 2
    RHS[:free_count] = [EA*(a/L)**3*(2*u/a+3*u*u/a/a+u*u*u/a/a/a) - lam*F]

    LHS[:free_count,-1] = -f[:free_count]
    LHS[-1,:] = [0.0, 1.0]
    RHS[-1] = lam-lam_hat
    print(LHS)
    print(RHS)
    return LHS, RHS 

# solve newton raphson
x = NewtonRaphson().Solve(CalculateSystem, x_initial=x)

