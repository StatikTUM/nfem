"""
NFEM teaching tool

A module for the non linear static analysis of 3D truss problems.
A light weight object oriented FEM code and some usefull postprocessing tools.
"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

from nfem.model import Model
from nfem.node import Node
from nfem.truss import Truss
from nfem.spring import Spring
from nfem.assembler import Assembler

from nfem.newton_raphson import newton_raphson_solve

from nfem.bracketing import bracketing

from nfem.visualization import *

import sys

IS_NOTEBOOK = 'ipykernel' in sys.modules

__all__ = [
    'IS_NOTEBOOK',
    'Model',
    'Node',
    'Truss',
    'Spring',
    'Assembler',
    'newton_raphson_solve',
    'bracketing',
    'info',
    'show_load_displacement_curve',
    'show_animation',
    'show_deformation_plot',
    'Plot2D',
]
