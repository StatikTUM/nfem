# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:07:11 2017

@author: Umer Sherdil Paracha
"""

import numpy as np
import NewtonRaphson as nr

def solve(pForce,pSteps,pResiduum,pStiffness,pFend,pRefDoF):
#    follows the load-displacement path of a structure using the load control method
#
#    Input variables:
#    pForce - reference force which is multiplied by lambda
#    pSteps - number of loadsteps until final force (pFend) is reached
#    pResiduum - maximum residuum allowed after correction steps
#    pStiffness - handle to a module which provides the functions computeStiffnessMatrix and computeResidual
#    pFend - norm of force which has to be reached at end of iterations
#
#    Output variables:
#    values - state of structure (u and lambda) when final force is reached

    print('Load control')
    global force # initialize globally available variable for reference force
    global lambdahat # initialize globally available variable for lambdahat
    global stiffness # initialize globally available variable for stiffness module
    global refDoF
    refDoF=pRefDoF
    stiffness = pStiffness # assign input variable to global variable
    lambdahat = np.linalg.norm(pFend)/(np.linalg.norm(pForce)*pSteps) # compute delta lambda for each solution step
    force = pForce # assign input variable to global variable

    nDof = len(force) # get number of dofs out of reference force

    values = np.zeros(nDof+1) # current state of system, start with zeros
    values_history=np.zeros([pSteps+1,nDof+1],float) # array to save all states during the solution steps
    values_history[0,:]=values # add initial state (is still zero, but code could be extended to use different initial conditions)

    counter=0 # counter for solution steps

    while(counter<pSteps):
        # print('LC Iteration ', counter+1)
        #print('')
        predictor = Prediction(values, force, lambdahat) # compute the predicted change of state
        delta=nr.solve(values, predictor, pResiduum, computeRHS) # compute the corrections which have to be made to predictor
        values = values + predictor + delta # add predictor and corrections to current state
        counter=counter+1 # increase iteration counter
        values_history[counter,:] = values # add next state to history

    return values_history # return final state

def Prediction(values, force, deltalambdahat):
    # computes predictor, which is the estimated difference between current state and next state

    K=stiffness.computeStiffnessMatrix(values,refDoF) # get stiffness matrix at current state
    Inverse=np.linalg.inv(K) # invert stiffness matrix
    prediction=np.dot(Inverse, force*deltalambdahat) # compute predicted change of u
    prediction=np.append(prediction, deltalambdahat) # add deltalambda to predictor
    return prediction

def computeRHS(values,predictor):
    # computes the right hand side of the equation system for NR
    # that is the residual vector and the constraint equation

    u = values[0:len(values)-1] # get current u
    deltau = predictor[0:len(values)-1] # get predicted change of u
    lam = values[len(values)-1] # get current lambda
    deltalam = predictor[len(values)-1] # get predicted change of lambda
    F = force * lam # compute current force
    deltaF = force * deltalam # compute predicted change of force
    R = stiffness.computeResidual(u+deltau,F+deltaF,refDoF) # compute residual
    C = Constraint(values,predictor) # compute constraint violation
    return np.append(R,C) # return vector with residual and constraint violation

def Constraint(values,predictor):
    return predictor[len(values)-1] - lambdahat # constraint for force control

