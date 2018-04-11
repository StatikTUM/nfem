# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:17:35 2017

@author: Lars Muth
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

def solve(values,pPredictor,residuum,computeRHS):
#    Solves an equation system, using Newton-Raphson method
#    Tries to find a solution, which makes the right hand side become 0
#
#    Input variables:
#    values - vector of last solution
#    predictor - difference vector from last solution to predicted solution
#    residuum - allowed residuum when NR iterations stop
#    computeRHS - function handle to a function, which returns right hand side of equation system
#
#    Output variables:
#    delta_sum - sum of all corrections which were made to the predictor

    predictor = copy.deepcopy(pPredictor)
    delta_sum = np.zeros(len(values)) # sum of corrections to predictor

#    value_history = np.array(predictor)
#    array which stores all predictors that were used during the NR iterations
#    only for debugging

    RHS = computeRHS(values,predictor) # compute initial RHS
#    counter = 0 # counter for NR iterations, only for debugging

    while (np.linalg.norm(RHS) > residuum): # check if specified residuum is reached
#        counter = counter+1 # increase iteration counter
#        print('NR Iteration ',counter) # print number of current iteration
        LHS = computeLHS(values,predictor,computeRHS) # compute LHS of equation system
        delta = np.dot(np.linalg.inv(LHS),RHS) # compute correction
        predictor += delta # add correction to predictor
        delta_sum += delta # sum up corrections
        RHS = computeRHS(values,predictor) # compute corrected RHS
#        value_history = np.vstack((value_history,predictor)) # append new predictor to history

#    print('Number of Newton-Raphson iterations: ',counter) # print needed iterations
#    print('')
#    if counter > 0:
#        plt.plot(value_history[:,0],value_history[:,1]) # plot correction steps
#        plt.show()
    return delta_sum # return sum of all corrections
#

def computeLHS(values, predictor, computeRHS):
#    computes the left hand side of the NR equation system
#    uses a central difference scheme to compute the jacobian of the RHS
#
#    Input variables:
#    values - vector of last solution
#    predictor - difference vector from last solution to predicted solution
#    computeRHS - function handle to a function, which returns right hand side of equation system
#
#    Output variables:
#    LHS - Jacobian matrix of the right hand side

    h = 0.000001 # step size for central difference
    dim = len(values) # dimensions of jacobian
    LHS = np.zeros((dim,dim)) # initialize matrix with zeros
    for i1 in range(dim): # for all variables (= rows of solution)
            pplus = copy.deepcopy(predictor) # copy predictor
            pplus[i1] = pplus[i1]+h # increase current row of predictor by step size
            pminus = copy.deepcopy(predictor) # copy again
            pminus[i1] = pminus[i1]-h # decrease current row of predictor by step size
            LHS[0:dim,i1] = np.transpose((computeRHS(values,pplus) - computeRHS(values,pminus)) / (2*h))
            # use central difference to compute derivative of RHS and paste it in LHS matrix
    return -LHS # return Jacobian