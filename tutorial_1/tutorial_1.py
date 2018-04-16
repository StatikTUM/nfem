"""
Main script for the Tutorial 1 for Lecture

Course :

Non-Linear Finite Element Anlaysis 
(https://campus.tum.de/tumonline/wbLv.wbShowLVDetail?pStpSpNr=950342272&pSpracheNr=2&pMUISuche=FALSE)

Lecturer: Prof. Dr.-Ing. Kai-Uwe Bletzinger
            Lehrstuhl für Statik
            Technische Universität München
            Arcisstr. 21
            D-80333 München 

Assistants: Armin Geiser M.Sc.  (armin.geiser@tum.de)
            and 
            Aditya Ghantasala M.Sc. (aditya.ghantasala@tum.de)
"""



"""
nfem : is the module for Non-Linear FEM where all the necessary tools and algorithms
are and will be implemented. 
"""
# add the path to the nfem tool to the PATH.
import sys
sys.path.append('..') 
# import necessary modules
from nfem import *

###########################################
# Create the FEM model
###########################################

# Creation of the model
model = Model('Two-Bar Truss')

model.AddNode(id='A', x=0, y=0, z=0)
model.AddNode(id='B', x=1, y=1, z=0)
model.AddNode(id='C', x=2, y=0, z=0)

model.AddTrussElement(id=1, node_a='A', node_b='B', youngs_modulus=1, area=1)
model.AddTrussElement(id=2, node_a='B', node_b='C', youngs_modulus=1, area=1)

model.AddSingleLoad(id='load 1', node_id='B', fv=-1)

model.AddDirichletCondition(node_id='A', dof_types='uvw', value=0)
model.AddDirichletCondition(node_id='B', dof_types='w', value=0)
model.AddDirichletCondition(node_id='C', dof_types='uvw', value=0)

###########################################
# 1:Linear analysis
###########################################

linear_model = model

# create a new model for each solution step
linear_model = linear_model.GetDuplicate() 
# define the load factor
linear_model.lam = 0.1
# perform a linear solution
linear_model.PerformLinearSolutionStep()

# create a new model for each solution step
linear_model = linear_model.GetDuplicate() 
# define the load factor
linear_model.lam = 0.2
# perform a linear solution
linear_model.PerformLinearSolutionStep()


###########################################
# 2:Non-Linear analysis
###########################################

non_linear_model = model

""" 
TODO : Add the Non linear model following the instructions in the Tutorial 
        And plot the PlotLoadDisplacementCurve for Nonlinear model
"""

plot = Plot2D()
plot.AddLoadDisplacementCurve(linear_model, dof=('B', 'v'))
plot.AddLoadDisplacementCurve(non_linear_model, dof=('B', 'v'))
plot.Show()
