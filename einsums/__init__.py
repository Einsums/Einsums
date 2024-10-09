"""
einsums
-------

This module allows for interaction with the C++ Einsums library.
"""

import sys
import os
import atexit

__modpath = os.path.dirname(__file__)

if __modpath not in sys.path :
    sys.path.append(__modpath)

try :
    from . import core
    if core.gpu_enabled() :
        from . import gpu_except
except (ModuleNotFoundError, ImportError) :
    try :
        import core
        if core.gpu_enabled() :
            import gpu_except
    except (ModuleNotFoundError, ImportError) as e :
        raise RuntimeError(f"File is {__file__}, path is {sys.path} and version is {sys.version}") from e

from . import utils # pylint: disable=wrong-import-position

core.initialize()

__outfile = False

def set_finalize_arg(out_arg : bool | str) :
    """
Sets the argument to pass to einsums.core.finalize. By default, the argument is False, which tells it not to print the
timing file. If the argument is True, the timing file will be printed at exit, and it will be called timings.txt. If
it is a file name, that will be used as the timing file, and the timing will be printed there on exit.
    """
    __outfile = out_arg

atexit.register(core.finalize, __outfile)