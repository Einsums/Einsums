"""
einsums
-------

This module allows for interaction with the C++ Einsums library.
"""

try :
    import core
except ModuleNotFoundError as e :
    import sys
    
    raise RuntimeError(f"Path is {sys.path} and version is {sys.version}") from e

from . import utils

core.initialize()
