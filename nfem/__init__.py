"""
NFEM teaching tool

A module for the non linear static analysis of 3D truss problems.
A light weight object oriented FEM code and some usefull postprocessing tools.
"""

import sys

__version__ = 'dev'

print(f"""
--------------------------------------------------------------------------------

                       _   ________________  ___
                      / | / / ____/ ____/  |/  /
                     /  |/ / /_  / __/ / /|_/ /
                    / /|  / __/ / /___/ /  / /
                   /_/ |_/_/   /_____/_/  /_/  Teaching Tool

  Authors:   Armin Geiser, Aditya Ghantasala, Thomas Oberbichler, Klaus Sautter
             Mahmoud Zidan
  Copyright: © 2018-2020 TUM Statik
  Version:   {__version__}

  This is a teaching tool! All results without warranty.

--------------------------------------------------------------------------------
""")

if sys.version_info < (3, 5):
    raise RuntimeError("The nfem module requires at least Python 3.5!")

from .model import Model
from .node import Node
from .truss import Truss
from .assembler import Assembler

from .newton_raphson import newton_raphson_solve
from .path_following_method import LoadControl
from .path_following_method import DisplacementControl
from .path_following_method import ArcLengthControl

from .visualization import interact
from .visualization import show_load_displacement_curve, show_animation, show_deformation_plot
from .visualization import Plot2D, Animation3D, DeformationPlot3D

from .bracketing import bracketing

def test():
    import unittest
    suite = unittest.TestLoader().discover('.', pattern='test*')
    unittest.TextTestRunner(verbosity=2).run(suite)
