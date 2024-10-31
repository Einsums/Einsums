"""
einsums
-------

This module allows for interaction with the C++ Einsums library.
"""

print("Before imports.", flush=True)

import sys
import os
import atexit

print("After system imports. Setting up module path.", flush=True)

__modpath = os.path.dirname(__file__)

if __modpath not in sys.path :
    sys.path.append(__modpath)

print("Set up the module path. Importing core.", flush=True)

try :
    from . import core
    print("Imported core. Checking GPU compatibility.", flush=True)
    if core.gpu_enabled() :
        print("Importing gpu_except.", flush=True)
        from . import gpu_except
        print("Imported GPU except.", flush=True)
except (ModuleNotFoundError, ImportError) :
    try :
        print("First try failed. Trying again.", flush=True)
        import core
        print("Imported core. Checking GPU compatibility.", flush=True)
        if core.gpu_enabled() :
            print("Importing gpu_except.", flush=True)
            import gpu_except
            print("Imported gpu_except.", flush=True)
    except (ModuleNotFoundError, ImportError) as e :
        raise RuntimeError(f"File is {__file__}, path is {sys.path} and version is {sys.version}") from e

print("Imported C++ libraries successfully. Importing Python modules.", flush=True)

print("Importing utils.", flush=True)
from . import utils # pylint: disable=wrong-import-position
print("Imported utils. Initializing.", flush=True)

core.initialize()

print('Initialized', flush=True)

__outfile = False

def set_finalize_arg(out_arg : bool | str) :
    """
Sets the argument to pass to einsums.core.finalize. By default, the argument is False, which tells it not to print the
timing file. If the argument is True, the timing file will be printed to standard output. If
it is a file name, that will be used as the timing file, and the timing will be printed there on exit.
    """
    __outfile = out_arg

print("Registering finalization.", flush=True)
atexit.register(core.finalize, __outfile)
print("Registered finalization. Everything finished.", flush=True)