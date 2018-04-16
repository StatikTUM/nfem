print("""

--------------------------------------------------------------------------------

                       _   ________________  ___
                      / | / / ____/ ____/  |/  /
                     /  |/ / /_  / __/ / /|_/ / 
                    / /|  / __/ / /___/ /  / /  
                   /_/ |_/_/   /_____/_/  /_/  Teaching Tool              

  Authors:   Armin Geiser, Aditya Ghantasala, Thomas Oberbichler, Klaus Sautter
  Copyright: © 2018 TUM Statik
  Version:   1.0

  This is a teaching tool! All results without warranty.

--------------------------------------------------------------------------------  
""")

from .model import Model
from .assembler import Assembler

from .newton_raphson import NewtonRaphson
from .path_following_method import LoadControl
from .path_following_method import DisplacementControl
from .path_following_method import ArcLengthControl
from .predictor import LoadIncrementPredictor
from .predictor import DisplacementIncrementPredictor
from .predictor import LastIncrementPredictor

from .visualization import Interact, ShowLoadDisplacementCurve, ShowHistoryAnimation, Plot2D