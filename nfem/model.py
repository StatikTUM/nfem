"""FIXME"""

from copy import deepcopy

import numpy as np
import numpy.linalg as la

from .node import Node
from .single_load import SingleLoad
from .truss import Truss

from .assembler import Assembler
from .newton_raphson import NewtonRaphson

from .path_following_method import LoadControl
from .path_following_method import DisplacementControl
from .predictor import LoadIncrementPredictor
from .predictor import DisplacementIncrementPredictor

class Model(object):
    """FIXME"""

    def __init__(self, name):
        """FIXME"""

        self.name = name
        self.nodes = dict()
        self.elements = dict()
        self.dirichlet_conditions = dict()
        self.neumann_conditions = dict()
        self.lam = 0.0
        self.previous_model = None

    def AddNode(self, id, x, y, z):
        """FIXME"""

        if id in self.nodes:
            raise RuntimeError('The model already contains a node with id {}'.format(id))

        self.nodes[id] = Node(id, x, y, z)

    def AddTrussElement(self, id, node_a, node_b, youngs_modulus, area):
        """FIXME"""

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_a not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_a))

        if node_b not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_b))

        self.elements[id] = Truss(id, self.nodes[node_a], self.nodes[node_b], youngs_modulus, area)

    def AddDirichletCondition(self, node_id, dof_types, value):
        """FIXME"""

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        for dof_type in dof_types:
            dof = (node_id, dof_type)

            if dof in self.dirichlet_conditions:
                raise RuntimeError('The model already contains a dirichlet condition for {}'.format(dof))

            self.dirichlet_conditions[dof] = value

    def AddSingleLoad(self, id, node_id, fu=0, fv=0, fw=0):
        """FIXME"""

        if id in self.elements:
            raise RuntimeError('The model already contains an element with id {}'.format(id))

        if node_id not in self.nodes:
            raise RuntimeError('The model does not contain a node with id {}'.format(node_id))

        self.elements[id] = SingleLoad(id, self.nodes[node_id], fu, fv, fw)

    def GetInitialModel(self):
        """FIXME"""

        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

        return current_model

    def GetModelHistory(self):
        """FIXME"""

        history = [self]

        current_model = self

        while current_model.previous_model is not None:
            current_model = current_model.previous_model

            history = [current_model] + history

        return history

    def GetDuplicate(self, name=None):
        """FIXME"""

        temp_previous_model = self.previous_model
        self.previous_model = None

        duplicate = deepcopy(self)

        self.previous_model = temp_previous_model
        duplicate.previous_model = self

        if name is not None:
            duplicate.name = name

        return duplicate

    def PerformLinearSolutionStep(self, lam=1.0):
        """Just for testing"""

        assembler = Assembler(self)

        self.lam = lam
        dof_count = assembler.dof_count

        u = np.zeros(dof_count)

        for dof, value in self.dirichlet_conditions.items():
            index = assembler.IndexOfDof(dof)
            u[index] = value

        k = np.zeros((dof_count, dof_count))
        f = np.zeros(dof_count)

        assembler.AssembleMatrix(k, lambda element: element.CalculateStiffnessMatrix())
        assembler.AssembleVector(f, lambda element: element.CalculateExternalForces())

        f *= self.lam

        free_count = assembler.free_dof_count

        a = k[:free_count, :free_count]
        b = f[:free_count] - k[:free_count, free_count:] @ u[free_count:]

        u[:free_count] = la.solve(a, b)

        for index, dof in enumerate(assembler.dofs):
            node_id, dof_type = dof

            value = u[index]

            self.nodes[node_id].Update(dof_type, value)

    def PerformNonLinearSolutionStep(self, 
                                     path_following_class=LoadControl, 
                                     predictor_class=LoadIncrementPredictor, 
                                     #path_following_class=DisplacementControl, 
                                     #predictor_class=DisplacementIncrementPredictor,
                                     prescribed_value=1.0 ):
        """Currently hardcoded for LoadControl"""

        path_following_method = path_following_class(prescribed_value)

        # create a model for the predictor
        predictor_model = self
        predictor_model.internal_flag = True

        # calculate the direction of the predictor
        predictor_model = predictor_class().Predict(predictor_model)

        # rotate the predictor if necessary (e.g. for branch switching)

        # scale the predictor so it fulfills the path following constraint
        predictor_model = path_following_method.ScalePredictor(predictor_model)

        # initialize working matrices and functions for newton raphson
        model = predictor_model
        assembler = Assembler(model)
        dof_count = assembler.dof_count
        free_count = assembler.free_dof_count

        k = np.zeros((dof_count,dof_count))
        f = np.zeros(dof_count)

        LHS = np.zeros((free_count+1, free_count+1))
        RHS = np.zeros(free_count+1)

        x = np.zeros(free_count+1) # TODO assemble from model: u = x-reference_x
        for i in range(free_count):
            print(i)
            dof = assembler.dofs[i]
            node = model.nodes[dof[0]]
            if dof[1] == 'u':
                x[i] = node.x - node.reference_x
            elif dof[1] == 'v':
                x[i] = node.y - node.reference_y
            elif dof[1] == 'w':
                x[i] = node.z - node.reference_z
        x[-1] = model.lam
        
        u = np.zeros(dof_count)

        # NOTE stiffness seems to be correct, problem with RHS or update?
        def CalculateSystem(x):
            print("x:", x)
            u[:free_count] = x[:-1]
            model.lam = x[-1]
            
            for index, dof in enumerate(assembler.dofs):
                node_id, dof_type = dof

                value = u[index]

                model.nodes[node_id].Update(dof_type, value)

            #initialize (set to zero)
            ke = np.zeros((dof_count,dof_count))
            ku = np.zeros((dof_count,dof_count))
            kg = np.zeros((dof_count,dof_count))
            f = np.zeros(dof_count)
            internal_f = np.zeros(dof_count)

            # assemble stiffness and force
            assembler.AssembleMatrix(ke, lambda element: element.CalculateElasticStiffnessMatrix())
            assembler.AssembleMatrix(ku, lambda element: element.CalculateGeometricStiffnessMatrix())
            k = ke + ku + kg
            assembler.AssembleVector(f, lambda element: element.CalculateExternalForces())
            assembler.AssembleVector(internal_f, lambda element: element.CalculateInternalForces())

            # assemble left and right hand side for newton raphson

            # mechanical system
            LHS[:free_count, :free_count] = k[:free_count, :free_count]
            LHS[-1,:free_count] = -f[:free_count]
            RHS[:free_count] = internal_f[:free_count]-f[:free_count]*model.lam

            # constraint
            LHS[-1,:] = path_following_method.CalculateDerivatives(model, LHS[-1,:])
            RHS[-1] = path_following_method.CalculateConstraint(model)
            print('LHS', LHS)
            print('RHS', RHS)
            return LHS, RHS 

        # solve newton raphson
        x = NewtonRaphson().Solve(CalculateSystem, x_initial=x)

        # update model (maybe this should happen already in newton raphson)
        # TODO

        # remove iterations from history
        # TODO

        #model.name = f'Non linear solution step (lambda={lam:.3})'
        return model
