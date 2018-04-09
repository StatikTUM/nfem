# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:57:36 2017

@author: Lars Muth
"""

import input2D as ip
#my_fileName = input("File name example: inputIFEM.txt\nFile Name: ");           #Dylan added
net = ip.readInput("inputIFEM.txt")#net = ip.readInput(my_fileName)

import numpy as np
np.set_printoptions(threshold=np.nan)
K_CurrentlyRegistered=np.zeros([2*net.Nodes.__len__(),2*net.Nodes.__len__()],float) #Dylan changed

def computeStiffnessMatrix(u,refDoF):
     K=AssembleMasterStiffOfExampleTruss()
#--------------------------------------------------------------------------------- Dylan added
#     kInput=[]
#     for i in range(0,net.Controls.getDoFList().__len__(),2):
#         kInput.append(i+1)
#         kInput.append(i+2)
#         if (net.Controls.getDoFList()[i] == 1 and net.Controls.getDoFList()[i+1] == 1):
#             del kInput[kInput.__len__()-1]
#             del kInput[kInput.__len__()-1]
     kInput=[]
     for i in range(0,net.Controls.getDoFList().__len__()):
         kInput.append(i+1)
         if (net.Controls.getDoFList()[i] == 1):
             del kInput[kInput.__len__()-1]
#---------------------------------------------------------------------------------
     K=ModifyMasterStiffForDBC(kInput,K)                                        #Dylan changed
#     RegisterCurrentStiffnessMatrix(K);
     K=K*f(u[refDoF-1])                                                                #DYLAN - why u[7]?
     return K

#DYLAN - the nonlinear force?
def f(x):
    return (x*x)+1

def FormElemStiff2DTwoNodeBar(xyi,xyj,E,A):
    #----------------------------------------------------------
    # Element Stiffness Function for 2D Two-Node Bar
    #----------------------------------------------------------
    # input:
    # xyi[2] ... x,y coordinates node i
    # xyj[2] ... x,y coordinates node j
    # E ... Young's modulus
    # A ... cross sectional area
    # output:
    # Ke[4,4] ... element stiffness matrix
    #----------------------------------------------------------
    dx = xyj[0] - xyi[0]
    dy = xyj[1] - xyi[1]
    L = np.sqrt(dx**2 + dy**2)

    c = dx / L
    cc = c**2
    s = dy / L
    ss = s**2
    cs = c * s

    T = np.matrix( ((cc, cs,-cc,-cs),
                     (cs, ss,-cs,-ss),
                     (-cc,-cs, cc, cs),
                     (-cs,-ss, cs, ss)) )

    Ke = T * E * A / L
    return Ke

def MergeElemIntoMasterStiff(Ke,eft,Kin):
    #----------------------------------------------------------
    # Merge element stiff matix into master matrix
    #----------------------------------------------------------
    # input:
    # Ke[4,4] ... element stiffness matrix
    # eft[4] ... element freedom table
    # Kin[n,n] ... master stiffness matrix, ndof = n
    # output:
    # Kin[n,n] ... modified master stiffness matrix
    #----------------------------------------------------------
    for i in range(0, 4):
        ii = eft[i]-1
        for j in range(i,4):
            jj = eft[j]-1

            Kin[ii,jj] = Kin[ii,jj] + Ke[i,j]
            Kin[jj,ii] = Kin[ii,jj]
            #print('Kin',Kin)
    return Kin

def ModifyMasterStiffForDBC(pdof,K):
    #----------------------------------------------------------
    # Modify master for displ. boundary conditions
    #----------------------------------------------------------
    # input:
    # pdof[npdof] ... npdof prescribed degrees of freedom
    # K[nk,nk] ... master stiffness matrix, ndof = nk
    # output:
    # Kmod[nk,nk] ... modified master stiffness matrix
    #----------------------------------------------------------
    nk = np.shape(K)[0]
    npdof = np.shape(pdof)[0]

    # ---------- copy master stiffness matrix -----------------
    Kmod = np.copy(K)
    # ---------- evaluate prescribed degrees of freedom--------
    for k in range(0, npdof):
        i = pdof[k]-1
    # ---------- clear rows and columns -----------------------
        for j in range(0, nk):
            Kmod[i,j] = 0
            Kmod[j,i] = 0
    # ---------- set diagonal to 1 ----------------------------
        Kmod[i,i] = 1

    return Kmod

def AssembleMasterStiffOfExampleTruss():
    #----------------------------------------------------------
    # Assembling master stiffness matrix
    #xyi,xyj,E,A    DYLAN - these are the variables for FormElemStiff2DTwoNodeBar
        # xyi[2] ... x,y coordinates node i
        # xyj[2] ... x,y coordinates node j
        # E ... Young's modulus
        # A ... cross sectional area
    #Ke,eft,Kin     DYLAN - these are the variables for MergeElemIntoMasterStiff
        # Ke[4,4] ... element stiffness matrix
        # eft[4] ... element freedom table
        # Kin[n,n] ... master stiffness matrix, ndof = n
    #----------------------------------------------------------
    K = np.zeros((net.Controls.getDoFList().__len__(),
                  net.Controls.getDoFList().__len__()))                         #create a zero matrix twice the number of nodes (aka. the number of local degrees of freedom)

    for i in range (net.Elements.__len__()):                                    #for each element create a stiffness matrix, and merge it into the master stiffness matrix
        Ke = FormElemStiff2DTwoNodeBar([net.Elements[i].nodeA.x,
                                        net.Elements[i].nodeA.y],
                                       [net.Elements[i].nodeB.x,
                                        net.Elements[i].nodeB.y],
                                        net.Elements[i].E,net.Elements[i].A)
        K = MergeElemIntoMasterStiff(Ke, [net.Elements[i].nodeA.ID*2-1,
                                          net.Elements[i].nodeA.ID*2,
                                          net.Elements[i].nodeB.ID*2-1,
                                          net.Elements[i].nodeB.ID*2 ],K)
    #--------------------------------------------------------------------------
    return K


def computeResidual(u,f,refDoF):
    return f-np.dot(computeStiffnessMatrix(u,refDoF),u)



