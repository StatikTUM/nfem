# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:57:36 2017

@author: Lars Muth
"""

import input3D as ip
#my_fileName = input("File name example: inputIFEM.txt\nFile Name: ");           #Dylan added
my_fileName = "inputIFEM3D.txt"
net = ip.readInput(my_fileName)

import numpy as np
np.set_printoptions(threshold=np.nan)
K_CurrentlyRegistered=np.zeros([2*net.Nodes.__len__(),2*net.Nodes.__len__()],float) #Dylan changed

def computeStiffnessMatrix(u,refDoF):
     K=AssembleMasterStiffOfExampleTruss()
#--------------------------------------------------------------------------------- Dylan added
     kInput=[]
     for i in range(0,net.Controls.getDoFList().__len__()):
         kInput.append(i+1)
         if (net.Controls.getDoFList()[i] == 1):
             del kInput[kInput.__len__()-1]
#---------------------------------------------------------------------------------
     K=ModifyMasterStiffForDBC(kInput,K)                                        #Dylan changed
#     RegisterCurrentStiffnessMatrix(K);
     K=K*f(u[refDoF-1])
     return K

#DYLAN - the nonlinear force?
def f(x):
    return (x*x)+1

def FormElemStiff3DTwoNodeBar(xyz_i,xyz_j,E,A):
    #----------------------------------------------------------
    # Element Stiffness Function for 3D Two-Node Bar
    #----------------------------------------------------------
    # input:
    # xyz_i[3] ... x,y,z coordinates node i
    # xyz_j[3] ... x,y,z coordinates node j
    # E ... Young's modulus
    # A ... cross sectional area
    # output:
    # Ke[6,6] ... element stiffness matrix
    #----------------------------------------------------------

    # Calculate Length
    dx = xyz_j[0] - xyz_i[0]
    dy = xyz_j[1] - xyz_i[1]
    dz = xyz_j[2] - xyz_i[2]
    length = np.sqrt( dx**2 + dy**2 + dz**2 )

    # Construct Transformation Matrix
    c_x = dx/length
    c_y = dy/length
    c_z = dz/length

    T = np.matrix( ((c_x**2,   c_x*c_y,   c_x*c_z,      -c_x**2,   -c_x*c_y,   -c_x*c_z),
                    (c_x*c_y,  c_y**2,    c_y*c_z,      -c_x*c_y,  -c_y**2,    -c_y*c_z),
                    (c_x*c_z,  c_y*c_z,   c_z**2,       -c_x*c_z,  -c_y*c_z,   -c_z**2),
                    (-c_x**2,   -c_x*c_y,   -c_x*c_z,    c_x**2,    c_x*c_y,    c_x*c_z),
                    (-c_x*c_y,  -c_y**2,    -c_y*c_z,    c_x*c_y,   c_y**2,     c_y*c_z),
                    (-c_x*c_z,  -c_y*c_z,   -c_z**2,     c_x*c_z,   c_y*c_z,    c_z**2)) )

    # Calculate Element Stiffness Matrix
    Ke = T * E * A / length
    return Ke

def Merge3DElemIntoMasterStiff(Ke,eft,Kin):
    #----------------------------------------------------------
    # Merge 3D element stiff matix into master matrix
    #----------------------------------------------------------
    # input:
    # Ke[6,6] ... element stiffness matrix
    # eft[6] ... element freedom table
    # Kin[n,n] ... master stiffness matrix, ndof = n
    # output:
    # Kin[n,n] ... modified master stiffness matrix
    #----------------------------------------------------------
    for i in range(0, 6):
        ii = eft[i]-1

        for j in range(i,6):
            jj = eft[j]-1

            Kin[ii,jj] = Kin[ii,jj] + Ke[i,j]
            Kin[jj,ii] = Kin[ii,jj]

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
    #xyzi,xyzj,E,A    DYLAN - these are the variables for FormElemStiff2DTwoNodeBar
        # xyzi[1,3] ... x,y,z coordinates node i
        # xyzj[1,3] ... x,y,z coordinates node j
        # E ... Young's modulus
        # A ... cross sectional area
    #Ke,eft,Kin     DYLAN - these are the variables for MergeElemIntoMasterStiff
        # Ke[6,6] ... element stiffness matrix
        # eft[6] ... element freedom table
        # Kin[n,n] ... master stiffness matrix, ndof = n
    #----------------------------------------------------------
    K = np.zeros((net.Controls.getDoFList().__len__(),
                  net.Controls.getDoFList().__len__()))                         #create a zero matrix twice the number of nodes (aka. the number of local degrees of freedom)

    for i in range (net.Elements.__len__()):                                    #for each element create a stiffness matrix, and merge it into the master stiffness matrix
        Ke = FormElemStiff3DTwoNodeBar([net.Elements[i].nodeA.x,
                                        net.Elements[i].nodeA.y,
                                        net.Elements[i].nodeA.z],
                                       [net.Elements[i].nodeB.x,
                                        net.Elements[i].nodeB.y,
                                        net.Elements[i].nodeB.z],
                                        net.Elements[i].E,net.Elements[i].A)
        K = Merge3DElemIntoMasterStiff(Ke, [net.Elements[i].nodeA.ID*3-2,
                                          net.Elements[i].nodeA.ID*3-1,
                                          net.Elements[i].nodeA.ID*3,
                                          net.Elements[i].nodeB.ID*3-2,
                                          net.Elements[i].nodeB.ID*3-1,
                                          net.Elements[i].nodeB.ID*3 ],K)
    #--------------------------------------------------------------------------
    return K

def computeResidual(u,f,refDoF):
    return f-np.dot(computeStiffnessMatrix(u,refDoF),u)
