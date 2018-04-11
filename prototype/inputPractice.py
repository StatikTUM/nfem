# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 23:52:32 2018

@author: baxte
"""
import numpy as np
# This script will read a text file which is in the following specific format
#------------------------------------------------------------------------------
#integer number of steps
#residuum
#displacement
#arclength
#load
#integer number of nodes
#integer number of elements
#nodeID x y z Px Py ux uy DoFx DofY
#repeat the above line for all nodes
#elementID nodeA nodeB E A alpha deltaT
#repeat the above line for all elements
#------------------------------
#eg. IFEM Exercise 4 - saved as inputIFEM.txt in same directory as current script
#------------------------------
#100
#0.000001
#-1
#0.0155
#-3.3
#6
#5
#1	0	0	0  0	0	0  0	0	0  1	0  1
#2	30	0	0  0	0	0  0	-2	0  1	0  1
#3	0	10	0  0	0	0  0	0	0  0	1  1
#4	10	10	0  0	-5	0  0	0	0  1	1  1
#5	20	10	0  0	-5	0  0	0	0  1	1  1
#6	30	10	0  0	0	0	0  0	0  0	1  1
#1	1	4	100	 1	0.001	0
#2	2	5	100	 1	0.001	0
#3	3	4	100	 1	0.001	30
#4	4	5	100	 1	0.001	30
#5	5	6	100	 1	0.001	30
#------------------------------------------------------------------------------

class network(object):
    #----------------------------------------------------------
    # Base class containing all variables for the system
    #----------------------------------------------------------
    # members: 
    #   Nodes, Elements, Controls (see descriptions in __init__)
    # output:
    #   
    #----------------------------------------------------------
    def __init__(self, Nodes, Elements, Controls):
        self.Nodes = Nodes                                                      #list containing multiple Node classes
        self.Elements = Elements                                                #list containing multiple Element classes
        self.Controls = Controls                                                #list containing all remaining calculation information
    
    def printNode(self, ID):
        print("Node",self.Nodes[ID-1].ID, 
              "[ x:",self.Nodes[ID-1].x,"y:",self.Nodes[ID-1].y,
              "Px:",self.Nodes[ID-1].Px,"Py:",self.Nodes[ID-1].Py,
              "ux:",self.Nodes[ID-1].ux,"uy:",self.Nodes[ID-1].uy,
              "DoFx:",self.Nodes[ID-1].DoFx,"DoFy:",self.Nodes[ID-1].DoFy, "]")
    
    def printElement(self, ID):
        print("Element",self.Elements[ID-1].ID, 
              "[ NodeA:",self.Elements[ID-1].nodeA.ID,"NodeB:",self.Elements[ID-1].nodeB.ID,
              "E:",self.Elements[ID-1].E,"A:",self.Elements[ID-1].A,
              "alpha:",self.Elements[ID-1].alpha,"deltaT:",self.Elements[ID-1].deltaT, "]")
    
    def printControls(self):
        print("Solver controls\nSteps: ",self.Controls.steps,"\nResiduum: ",
              self.Controls.residuum,"\nEnd displacement: ",
              self.Controls.uEnd,"\nArclength: ",
              self.Controls.arclength,"\nEnd Load: ",
              self.Controls.fendVal,"\n\nNumber of nodes: ",
              self.Controls.nNodes,"\n[#, x, y, Px, Py, ux, uy, DoFx, DoFy]")
        print("\nNumber of elements: ",self.Elements.__len__(),"\n[#, nodeA, nodeB, E, A, alpha, deltaT]")            
       
class Node(object):
    #----------------------------------------------------------
    # Containing all variables for a Node
    #----------------------------------------------------------
    # members: 
    #   ID, x, y, Px, Py, ux, uy, DoFx, DoFy (see descriptions in __init__)
    # output:
    #   
    #----------------------------------------------------------
    def __init__(self, ID, x, y, Px, Py, ux, uy, DoFx, DoFy):
        self.ID = ID                                                            #unique identification integer    
        self.x = x                                                              #x coordinate
        self.y = y                                                              #y coordinate
        self.Px = Px                                                            #external force in x direction
        self.Py = Py                                                            #external force in y direction
        self.ux = ux                                                            #pre-displacement in x direction
        self.uy = uy                                                            #pre-displacement in y direction    
        self.DoFx = DoFx                                                        #degree of freedom in x, 1=free, 0=constrained
        self.DoFy = DoFy                                                        #degree of freedom in y, 1=free, 0=constrained    
 
class Element(object):
    #----------------------------------------------------------
    # Containing all variables for an Element
    #----------------------------------------------------------
    # members: 
    #   ID, nodeA, nodeB, E, A, alpha, deltaT (see descriptions in __init__)
    # output:
    #   calcLength                      ...     returns the length of the element using nodeA and nodeB coordinates
    #   getSingleElementFreedomTable    ...     returns the ID number of the element's degree of freedom for the master stiffness matrix 
    #----------------------------------------------------------
    def __init__(self, ID, nodeA, nodeB, E, A, alpha, deltaT):
        self.ID = ID                                                            #unique identification integer    
        self.nodeA = nodeA                                                      #Node class at base of Element
        self.nodeB = nodeB                                                      #Node class at tip of Element
        self.E = E                                                              #Youngs modulus
        self.A = A                                                              #cross sectional area
        self.alpha = alpha                                                      #coefficient of thermal expansion
        self.deltaT = deltaT                                                    #change in temperature experienced
    
    def calcLength(self):
        deltaX = self.nodeA.x-self.nodeB.x
        deltaY = self.nodeA.y-self.nodeB.y
        return np.sqrt(deltaX*deltaX+deltaY*deltaY)
    def printLength(self):
        print("Element",self.ID,"length:",self.calcLength())
class Controls(object):
    #----------------------------------------------------------
    # Containing all calculation control parameters
    #----------------------------------------------------------
    # members: 
    #   steps, residuum, uEnd, uNode, uDirection, arclength, fendVal, nodeList (see descriptions in __init__)
    # output:
    #   getForce    ...     returns an array which contains all external forces
    #   getDoF      ...     returns the integer DoF value for the whole network
    #   getDoFList  ...     returns a list of DoFs mapped according to the Node ID
    #   getLoadEnd  ...     returns an array which contains the end-load-control in the necessary directions at the necessary Nodes
    #----------------------------------------------------------
    def __init__(self, steps, residuum, uEnd, uNode, uDirection, arclength, fendVal, nodeList):
        self.steps = steps                                                      #maximum number of steps for all solver methods
        self.residuum = residuum                                                #      
        self.uEnd = uEnd                                                        #final displacement for displacement-control method
        self.uNode = uNode
        self.uDirection = uDirection
        self.arclength = arclength                                              #arclength for arclength-control method        
        self.fendVal = fendVal                                                  #final force for load-control method
        self.nodeList = nodeList                                                #list of all nodes in network
    
    def getForce(self):
        force=np.empty(self.nodeList.__len__()*2)
        for i in range(self.nodeList.__len__()):
            force[i*2]=(self.nodeList[i].Px)                                    #copy x direction forces to the force array
            force[i*2+1]=(self.nodeList[i].Py)                                  #copy y direction force to the force array
        return force
    def printForce(self):
        print("System Force:",self.getForce())
    
    def getDoF(self): #DYLAN: im not sure what is the best value to return, the dc.solve uses the DoF ID to solve until the set max displacement at that node is reached 
#        for i in range(self.nodeList.__len__()):
#            dof = dof+self.nodeList[i].DoFx+self.nodeList[i].DoFy               #calculate system degree of freedom
        i=0
        while (i==0):
            if (self.uDirection == 1 or self.uDirection == 2):
                dof=self.uNode*2-2+self.uDirection
                break
            else:
                print("DoF direction")
                print(self.uDirection)
                print("not recognized.\nPlease enter the Displacement Solver DoF number: eg. 1=x, 2=y, etc.")
                self.uDirection=int(input())
        return dof
    def printDoF(self):
        print("System DoF:",self.getDoF())
        
    def getDoFList(self):                                                       #assemble a list of the DoFs
        listed=[]
        for i in range(self.nodeList.__len__()):
            listed.append(self.nodeList[i].DoFx)
            listed.append(self.nodeList[i].DoFy)
        return listed
    def printDoFList(self):
        print("System DoF List:",self.getDoFList())
        
    def getLoadEnd(self):
        fend=np.empty(self.nodeList.__len__()*2)
        for i in range(self.nodeList.__len__()*2):
            if (self.getForce()[i]!=0):
                fend[i]=self.fendVal                                            #copy end load control value to correct position in the load array
        return fend
    def printLoadEnd(self):
        print("System end load:", self.getLoadEnd())

def readInput(fileName):
    #----------------------------------------------------------
    # Read the text file given in the specific format
    #----------------------------------------------------------
    # input:
    # string("fileName.txt") ... the name of the input file
    # output:
    # my_network ... class containing all input file information in a generic and retrievable format
    #----------------------------------------------------------
    f = open(fileName,'r')                                                      #opens the text file
    steps = int(f.readline())                                                   #reads number of steps for all solver methods
    residuum = float(f.readline())                                              #reads residuum for all solver methods
    uEnd = float(f.readline())                                                  #reads solver control for displacement method
    uNode = int(f.readline())
    uDirection = int(f.readline())
    arclength = float(f.readline())                                             #reads solver control for arclength method
    fendVal = float(f.readline())                                               #reads solver control for load method
    nNodes = int(f.readline())                                                  #reads the number of nodes
    nElements = int(f.readline())                                               #reads the number of elements
    readList=[]
    nodeList=[]
    for i in range(nNodes):
        readList = f.readline().split()
        #ID, x, y, Px, Py, ux, uy, DoFx, DoFy                                   #adds nodal information to the list of nodes
        nodeList.append(Node(int(readList[0]), float(readList[1]), 
                             float(readList[2]), float(readList[3]), 
                             float(readList[4]), float(readList[5]), 
                             float(readList[6]), int(readList[7]), 
                             int(readList[8])))
    elementList=[]    
    for i in range(nElements):
        readList = f.readline().split()
        #ID, nodeA, nodeB, E, A, alpha, deltaT                                  #adds elemental information to the list of elements
        elementList.append(Element(int(readList[0]), 
                                   nodeList[int(readList[1])-1], 
                                   nodeList[int(readList[2])-1], 
                                   float(readList[3]), float(readList[4]), 
                                   float(readList[5]), float(readList[6])))      
    f.close()                                                                   #close the text file
    return network(nodeList, elementList,
                    Controls(steps,residuum,uEnd,uNode,uDirection,arclength,fendVal,nodeList))   #write node, element, and control lists to a network object                 
#------------------------------------------------------------------------------  
