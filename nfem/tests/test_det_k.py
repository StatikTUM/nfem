import sys
sys.path.append('..') 

# import necessary modules
from nfem import *
import test_two_bar_truss_model

import math

# two bar truss model
model = test_two_bar_truss_model.get_model()

#======================================
# initial model
#======================================
model.solve_det_k()
assert(math.isclose(model.det_k, 0.4999999999999997, rel_tol=1e-12) )

#======================================
# after first solution step
#======================================
model.lam = 0.1
model.perform_non_linear_solution_step(strategy="load-control")
model.solve_det_k()
assert(math.isclose(model.det_k, 0.19510608810631772, rel_tol=1e-12) )
