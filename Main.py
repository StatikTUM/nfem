# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:03:14 2017

@author: Lars Muth
Need to address:
    1. force recovery
    2. check if temperature force is being calculated
    3. fix displacement boundary condition
    4. generalise the non-linear solver script

    Need to do:
        2. which dof is being investigated in plot
        3. user is provided with the possibility to input with graphical
        feedback like platesTool, make platesTool create the .txt file
        4. user input to decide which solver (one or multiple)
        5. use .json files as the input instead of .txt

"""

import numpy as np
import LoadControl as lc
import DispControl as dc
import ArclengthControl as ac
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#------------------------------------------------------------------------------
def plotStructure(figNum, nodes, u, EFT, noPlots, scalFac, dimensions):
    #----------------------------------------------------------
    # Plot the physical structure before, halfway through, and after loading
    #----------------------------------------------------------
    #input:
    #   nodes       ...     original node coordinates
    #   u           ...     displacement results from solver
    #   EFT         ...     complete element freedom table
    #   noPlots     ...     number of plots in figure
    #   scalFac     ...     scaling factor
    #   dimensions  ...     number of figure axis
    #----------------------------------------------------------
    x = np.zeros(len(nodes))
    if dimensions > 1:
        y = np.zeros(len(nodes))
    if dimensions > 2:
        z = np.zeros(len(nodes))
        fig = plt.figure(figNum, figsize=(figSize, figSize))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(noPlots):
            for k in range(len(nodes)):
                x[k] = nodes[k,0] + i/noPlots*u[dimensions*k]*scalFac
                if dimensions > 1:
                    y[k] = nodes[k,1] + i/noPlots*u[dimensions*k+1]*scalFac
                if dimensions > 2:
                    z[k] = nodes[k,2] + i/noPlots*u[3*k+2]*scalFac
            if i == 0:
                for k in range(len(EFT)):
                    ax.plot([x[EFT[k,0]-1],x[EFT[k,1]-1]],
                            [y[EFT[k,0]-1],y[EFT[k,1]-1]],
                            [z[EFT[k,0]-1],z[EFT[k,1]-1]],
                            'bo-',label='Initial Configuration',lw=2)
            elif i == noPlots-1:
                for k in range(len(EFT)):
                    ax.plot([x[EFT[k,0]-1],x[EFT[k,1]-1]],
                            [y[EFT[k,0]-1],y[EFT[k,1]-1]],
                            [z[EFT[k,0]-1],z[EFT[k,1]-1]],
                            'ro-',label='Final Configuration',lw=2)
            else:
                for k in range(len(EFT)):
                    ax.plot([x[EFT[k,0]-1],x[EFT[k,1]-1]],
                            [y[EFT[k,0]-1],y[EFT[k,1]-1]],
                            [z[EFT[k,0]-1],z[EFT[k,1]-1]],'kx--')
    else:
        fig = plt.figure(figNum, figsize=(figSize, figSize))
        for i in range(noPlots):
            for k in range(len(nodes)):
                x[k] = nodes[k,0] + i/noPlots*u[dimensions*k]*scalFac
                if dimensions > 1:
                    y[k] = nodes[k,1] + i/noPlots*u[dimensions*k+1]*scalFac
                if dimensions > 2:
                    z[k] = nodes[k,2] + i/noPlots*u[3*k+2]*scalFac
            if i == 0:
                for k in range(len(EFT)):
                    plt.plot([x[EFT[k,0]-1],x[EFT[k,1]-1]],
                             [y[EFT[k,0]-1],y[EFT[k,1]-1]],
                             'bo-',label='Initial Configuration',lw=2)
            elif i == noPlots-1:
                for k in range(len(EFT)):
                    plt.plot([x[EFT[k,0]-1],x[EFT[k,1]-1]],
                             [y[EFT[k,0]-1],y[EFT[k,1]-1]],
                             'ro-',label='Final Configuration',lw=2)
            else:
                for k in range(len(EFT)):
                    plt.plot([x[EFT[k,0]-1],x[EFT[k,1]-1]],
                             [y[EFT[k,0]-1],y[EFT[k,1]-1]],'kx--')

    plt.margins(0.1,0.1) # add 10% margin
    plt.title('Structural Deformation',fontsize=fontSize)
    plt.xlabel('x',fontsize=fontSize)
    plt.ylabel('y',fontsize=fontSize)
#    plt.legend(loc='best')
    plt.savefig('Deformation.png',bbox_inches='tight')
#------------------------------------------------------------------------------


#Ask user for 2D or 3D
i=0
while i==0:
    print("Dimension of system: 2 or 3")
    dimension=int(input())
    if (dimension==2 or dimension==3):
        break
    else:
        print("Incorrect user entry")
#IFEM example
if (dimension==2):
    import stiffnessmatrix_IFEM_truss as stiffness
else:
    import stiffnessmatrix_IFEM_truss3D as stiffness

##NFEM Example
#import stiffnessmatrix_NFEM_truss_engstrain as stiffness
#force = np.array([-1.0],'float') # NFEM example
#fend = np.array([0.5],'float') #dylanAdded
#dof = 1
#uend = 1.5
#steps = 200
#residuum = 0.000001
#arclength = 0.0155

#Calculations begin
force = stiffness.net.Controls.getForce()
steps = stiffness.net.Controls.steps
residuum = stiffness.net.Controls.residuum
arclength = stiffness.net.Controls.arclength
endLoad = stiffness.net.Controls.getLoadEnd()
endDisplacement = stiffness.net.Controls.uEnd
refDoF = stiffness.net.Controls.getDoF()

ac_results = ac.solve(force,steps,residuum,stiffness,arclength,refDoF)
lc_results = lc.solve(force,steps,residuum,stiffness,endLoad,refDoF)
dc_results = dc.solve(force,steps,residuum,stiffness,endDisplacement,refDoF)

##Output
figNum=1
fontSize=10
figSize=10
numdofs = len(stiffness.net.Controls.getForce())
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : fontSize}
plinewidth = 10
plt.rc('font', **font)
for printDoF in range(numdofs+1):
    plt.figure(figNum, figsize=(figSize, figSize))
    plt.plot(ac_results[:,printDoF],ac_results[:,numdofs],'*',markersize=2*plinewidth) # plot state history
    plt.plot(lc_results[:,printDoF],lc_results[:,numdofs],'o',markersize=plinewidth) # plot state history, Load factor will never exceed 1
    plt.plot(dc_results[:,printDoF],dc_results[:,numdofs],'x',markersize=plinewidth) # plot state history
    plt.xlabel('Displacement',fontsize=fontSize)
    plt.ylabel('Load factor',fontsize=fontSize)
    plt.legend(['Arc length control','Load control','Displacement control'])
    plt.title('Load-displacement curve for DoF %i' %(printDoF+1),fontsize=fontSize)
    plt.gca().invert_xaxis()
    plt.savefig('LD_curve.png',bbox_inches='tight')
    figNum=figNum+1

#Dylan changed the outputs to work for any list of nodes
#------------------------------------------------------------------------------
nodeCoords=np.empty([stiffness.net.Nodes.__len__(),dimension])
if (dimension==2):
    for i in range(stiffness.net.Nodes.__len__()):
        nodeCoords[i][0] = stiffness.net.Nodes[i].x
        nodeCoords[i][1] = stiffness.net.Nodes[i].y
else:
    for i in range(stiffness.net.Nodes.__len__()):
        nodeCoords[i][0] = stiffness.net.Nodes[i].x
        nodeCoords[i][1] = stiffness.net.Nodes[i].y
        nodeCoords[i][2] = stiffness.net.Nodes[i].z

ConnectionTable = np.empty([stiffness.net.Elements.__len__(),2], dtype=int)
for i in range(stiffness.net.Elements.__len__()):
    ConnectionTable[i][0] = stiffness.net.Elements[i].nodeA.ID
    ConnectionTable[i][1] = stiffness.net.Elements[i].nodeB.ID
#------------------------------------------------------------------------------
plotStructure(figNum,nodeCoords,ac_results[stiffness.net.Controls.steps,:],ConnectionTable,3,1,dimension) #Dylan generalise so that 2D and 3D work

plt.show()