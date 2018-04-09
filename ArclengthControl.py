# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:07:11 2017

@author: Umer Sherdil Paracha
"""

import numpy as np
import NewtonRaphson as nr

def solve(pForce,pSteps,pResiduum,pStiffness,pArclength,pRefDoF):
#    follows the load-displacement path of a structure using the arclength control method
#        
#    Input variables:
#    pForce - reference force which is multiplied by lambda
#    pSteps - number of loadsteps until final displacement (pUend) is reached
#    pResiduum - maximum residuum allowed after correction steps
#    pStiffness - handle to a module which provides the functions computeStiffnessMatrix and computeResidual
#    pArclength - arclength which shall be used
#    
#    Output variables:
#    values - state of structure (u and lambda) after last iteration
    
    print('Arc length control')
    global force
    global arclength
    global stiffness
    global refDoF
    refDoF = pRefDoF
    stiffness = pStiffness
    arclength = pArclength
    force = pForce
    
    nDof = len(force)
        
    values = np.zeros(nDof+1)
    values_history=np.zeros([pSteps+1,nDof+1],float) 
    values_history[0,:]=values
    
    counter=0
    
    while(counter<pSteps):
        # print('LC Iteration ', counter+1)
        #print('')
        predictor = Prediction(values, force, arclength)
        delta=nr.solve(values, predictor, pResiduum, computeRHS)
        values = values + predictor + delta
        counter=counter+1
        values_history[counter,:] = values

    return values_history

def Prediction(values,force, lambdahat):
    K=stiffness.computeStiffnessMatrix(values,refDoF)
    Inverse=np.linalg.inv(K)
    
    neg = 1
    if (np.linalg.det(K) < 0):
        neg = -1
        
    predictor=np.dot(Inverse, force*neg*lambdahat)
    predictor=np.append(predictor, neg*lambdahat)
    
    v = np.dot(Inverse,force)
    fn = np.sqrt(1+np.dot(v,v))
    
    scaling = arclength/((np.dot(v,predictor[0:len(predictor)-1])+predictor[len(predictor)-1])/fn)
    predictor = predictor * scaling * neg
    return predictor

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
    deltau = predictor[0:len(values)-1]
    deltalam = predictor[len(values)-1]
    
    v = np.dot(np.linalg.inv(stiffness.computeStiffnessMatrix(values+predictor,refDoF)),force)
    fn = np.sqrt(1+np.dot(v,v))
                     
    return np.abs(np.dot(v,deltau)+deltalam)/fn - arclength
    
