"""
einsums
-------

This module allows for interaction with the C++ Einsums library.
"""

import sys
import os

__modpath = os.path.dirname(__file__)

if __modpath not in sys.path :
    sys.path.append(__modpath)

try :
    from . import core
except (ModuleNotFoundError, ImportError) :
    try :
        import core
    except (ModuleNotFoundError, ImportError) as e :
        raise RuntimeError(f"File is {__file__}, path is {sys.path} and version is {sys.version}") from e

from . import utils # pylint: disable=wrong-import-position

core.initialize()
# The finalize method has already been registered in the C++ side.
