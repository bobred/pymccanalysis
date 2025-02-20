import sys

__version__ = '1.0.0'
__version_info__ = (1, 0, 0)

# check python version
if sys.version_info[0] < 3 or sys.version_info[1] < 9:
    raise ValueError("only supported is Python 3.9+. Please update your environment.")

from .wtscans import *
from .readmcc import *
from .detector_array import *
from .gamma_analysis import *

__all__ = ["Gamma", "XyProfile", "PDD", "DetectorArray"]
