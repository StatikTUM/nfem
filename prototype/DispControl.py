# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:07:11 2017

@author: Umer Sherdil Paracha
"""

import numpy as np
import NewtonRaphson as nr

def solve(pForce,pSteps,pResiduum,pStiffness,pUend,pRefDof):
#    follows the load-displacement path of a structure using the displacement control method
#        
#    Input variables:
#    pForce - reference force which is multiplied by lambda
#    pSteps - number of loadsteps until final displacement (pUend) is reached
#    pResiduum - maximum residuum allowed after correction steps
#    pStiffness - handle to a module which provides the functions computeStiffnessMatrix and computeResidual
#    pUend - displacement which has to be reached at end of iterations
#    pDof - degree of freedom at which the displacement shall be reached
#    
#    Output variables:
#    values - state of structure (u and lambda) when final displacement is reached
    
    print('Displacement control')
    global force # initialize globally available variable for reference force
    global stiffness # initialize globally available variable for stiffness module
    global uhat # initialize globally available variable for u increment
    global refDoF # initialize globally available variable for control dof
    stiffness = pStiffness  # assign input variable to global variable
    uhat = pUend/pSteps # compute delta lambda for each solution step
    refDoF = pRefDof  # assign input variable to global variable
    force = pForce  # assign input variable to global variable

    nDof = len(force)
        
    values = np.zeros(nDof+1)
    values_history=np.zeros([pSteps+1,nDof+1],float) 
    values_history[0,:]=values
    
    counter=0
    
    while(counter<pSteps):
        # print('LC Iteration ', counter+1)
        #print('')
        predictor = Prediction(values)
        delta=nr.solve(values, predictor, pResiduum, computeRHS)
        values += predictor + delta
        counter=counter+1
        values_history[counter,:] = values

    return values_history

def Prediction(values):
    lambdahat = 0.1
    
    K=stiffness.computeStiffnessMatrix(values,refDoF)
    Inverse=np.linalg.inv(K)
    
    u_estimate=np.dot(Inverse,force*lambdahat)
    scaling = uhat/u_estimate[refDoF-1]
    u_corrected = scaling*u_estimate
    force_corrected = np.dot(K,u_corrected)
    
    neg = 1.0
    if np.linalg.det(K) < 0:
        neg = -1.0
    lambdahat = neg * np.linalg.norm(force_corrected)/np.linalg.norm(force)
    
    prediction=np.append(u_corrected,lambdahat)
    return prediction

def computeRHS(values,predictor):
    u = values[0:len(values)-1]
    deltau = predictor[0:len(values)-1]
    lam = values[len(values)-1]
    deltalam = predictor[len(values)-1]
    F = force * lam
    deltaF = force * deltalam
    R = stiffness.computeResidual(u+deltau,F+deltaF,refDoF)
    C = Constraint(values,predictor)
    return np.append(R,C)

def Constraint(values,predictor):
    return predictor[refDoF-1] - uhat
    
