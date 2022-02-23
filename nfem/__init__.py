"""NFEM teaching tool.

A module for the non linear static analysis of 3D truss problems.
A light weight object oriented FEM code and some usefull postprocessing tools.
"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata  # type: ignore

__version__ = importlib_metadata.version(__name__)

from nfem.model import Model
from nfem.node import Node
from nfem.truss import Truss
from nfem.spring import Spring
from nfem.assembler import Assembler

from nfem.bracketing import bracketing

from nfem.visualization import Plot2D

__all__ = [
    'Model',
    'Node',
    'Truss',
    'Spring',
    'Assembler',
    'bracketing',
    'Plot2D',
]
