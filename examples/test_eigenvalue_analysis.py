import sys
sys.path.append('..') 

# import necessary modules
from nfem import *
import test_two_bar_truss_model

import math

# two bar truss model
model = test_two_bar_truss_model.get_model()

#======================================
# limit point
#======================================
limit_model = model.get_duplicate()
limit_model.lam = 0.1
limit_model.perform_non_linear_solution_step(strategy="load-control")
limit_model.solve_eigenvalues()

assert(math.isclose(limit_model.first_eigenvalue, 3.6959287916726304, rel_tol=1e-12) )
# TODO test eigenvector [0.0, 1.0]

#======================================
# bifurcation point
#======================================
bifurcation_model = model.get_duplicate()
bifurcation_model._nodes['B'].reference_y = 3.0
bifurcation_model._nodes['B'].y = 3.0
bifurcation_model.lam = 0.1
bifurcation_model.perform_non_linear_solution_step(strategy="load-control")
bifurcation_model.solve_eigenvalues()

assert(math.isclose(bifurcation_model.first_eigenvalue, 1.7745968576086002, rel_tol=1e-12) )
# TODO test eigenvector [1.0, 0.0]