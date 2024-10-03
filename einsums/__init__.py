"""
einsums
-------

This module allows for interaction with the C++ Einsums library.
"""

import sys

try :
    from . import core
except (ModuleNotFoundError, ImportError) :
    try :
        import core
    except (ModuleNotFoundError, ImportError) as e :
        raise RuntimeError(f"Path is {sys.path} and version is {sys.version}") from e

from . import utils

core.initialize()
