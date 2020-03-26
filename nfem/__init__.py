"""
NFEM teaching tool

A module for the non linear static analysis of 3D truss problems.
A light weight object oriented FEM code and some usefull postprocessing tools.
"""

__version__ = 'dev'

from .model import Model
from .node import Node
from .truss import Truss
from .assembler import Assembler

from .newton_raphson import newton_raphson_solve
from .path_following_method import LoadControl
from .path_following_method import DisplacementControl
from .path_following_method import ArcLengthControl

from .bracketing import bracketing

from .visualization import *


def info():
    print(f'--------------------------------------------------------------------------------')
    print(f'                                                                                ')
    print(f'                      _   ________________  ___                                 ')
    print(f'                     / | / / ____/ ____/  |/  /                                 ')
    print(f'                    /  |/ / /_  / __/ / /|_/ /                                  ')
    print(f'                   / /|  / __/ / /___/ /  / /                                   ')
    print(f'                  /_/ |_/_/   /_____/_/  /_/  Teaching Tool                     ')
    print(f'                                                                                ')
    print(f'  Authors:   Armin Geiser, Aditya Ghantasala, Thomas Oberbichler, Klaus Sautter ')
    print(f'             Mahmoud Zidan                                                      ')
    print(f'  Copyright: © 2018-2020 TUM Statik                                             ')
    print(f'  Version:   {__version__}                                                      ')
    print(f'                                                                                ')
    print(f'  This is a teaching tool! All results without warranty.                        ')
    print(f'                                                                                ')
    print(f'--------------------------------------------------------------------------------')


__all__ = [
    'Model',
    'Node',
    'Truss',
    'Assembler',
    'newton_raphson_solve',
    'LoadControl',
    'DisplacementControl',
    'ArcLengthControl',
    'bracketing',
    'info',
    'show_load_displacement_curve',
    'show_animation',
    'show_deformation_plot',
    'Plot2D',
    'Animation3D',
    'DeformationPlot3D',
]

try:
    from .visualization import interact
    __all__ += [
        'interact',
    ]
except Exception as _:
    pass
