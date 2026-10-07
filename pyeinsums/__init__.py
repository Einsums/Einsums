# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
einsums
-------

This module allows for interaction with the C++ Einsums library.
"""

import sys
import os
import atexit

import inspect
import datetime
import typing


def log(level: int, msg: str, always_print: bool=False, stack_info: typing.Optional[inspect.FrameInfo]=None):
    """
Log a message with a given level of urgency. Level 0 is for trace statements, 1 is for debugging statements,
2 is for general information, 3 is for recoverable warnings, 4 is for unrecoverable errors, 5 is for critical
errors.

:param level: The level to log at.
:param msg: The message to print.
:param always_print: If false, the logger will only print if the level parameter is greater than or equal to the
global log level limit. If true, the global log level limit is ignored.
:param stack_info: A ``FrameInfo`` object or ``None``, which tells the logger the function that raised the message.
If none, then the logger will use the frame info of the function that called it.
    """
    # Add 3 zeros to the end so it lines up with the C++ side's nanosecond time.
    timestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f000")
    
    # This enables colors on Windows terminals, according to the internet.
    os.system("")
    
    if stack_info is None:
        stack_info = inspect.stack()[1]
        
    global_log_level = 3
    
    try:
        global_log_level = core.GlobalConfigMap.get_singleton().get_int("log-level")
    except: 
        pass
    
    if level >= global_log_level or always_print:
        print(f"\033[0m[{timestr}] [einsums]", end=" ")
        
        match(level):
            case 0:
                print("[trace   ]", end=" ")
            case 1:
                print("[\033[36mdebug   \033[0m]", end=" ")
            case 2:
                print("[\033[32minfo    \033[0m]", end=" ")
            case 3:
                print("[\033[1;33mwarning \033[0m]", end=" ")
            case 4:
                print("[\033[1;31merror   \033[0m]", end=" ")
            case _:
                print("[\033[1;41mcritical\033[0m]", end=" ")
        print(f"[{os.path.basename(stack_info.filename)}:{stack_info.lineno}/{stack_info.function}] {msg}")


# Set the EINSUMS_DEBUG_IMPORT environment variable to debug the import process.
if "EINSUMS_DEBUG_IMPORT" in os.environ:
    __affirmative = ["on", "yes", "true", "1"]
    __negative = ["off", "no", "false", "0"]
    
    if os.environ["EINSUMS_DEBUG_IMPORT"].lower() in __affirmative:
        __import_log_print = True
    elif os.environ["EINSUMS_DEBUG_IMPORT"].lower() in __negative:
        __import_log_print = False
    del __affirmative
    del __negative


def log_trace(msg: str, always_print: bool=False):
    log(0, msg, always_print, inspect.stack()[1])

    
def log_debug(msg: str, always_print: bool=False):
    log(1, msg, always_print, inspect.stack()[1])

    
def log_info(msg: str, always_print: bool=False):
    log(2, msg, always_print, inspect.stack()[1])


def log_warn(msg: str, always_print: bool=False):
    log(3, msg, always_print, inspect.stack()[1])


def log_error(msg: str, always_print: bool=False):
    log(4, msg, always_print, inspect.stack()[1])


def log_critical(msg: str, always_print: bool=False):
    log(5, msg, always_print, inspect.stack()[1])


__modpath = os.path.dirname(__file__)

if __modpath not in sys.path:
    sys.path.append(__modpath)

try:
    from . import core
except (ModuleNotFoundError, ImportError):
    try:
        print("Importing core in a different way.")
        import core
    except (ModuleNotFoundError, ImportError) as e:
        raise RuntimeError(
            f"File is {__file__}, path is {sys.path} and version is {sys.version}"
        ) from e

from . import utils  # pylint: disable=wrong-import-position


def initialize():
    """
    Filter out Python arguments and pass on einsums arguments. Einsums arguments are prefixed with
    '--einsums'.
    """
    pass_args = [sys.argv[0]]

    if len(sys.argv) > 1:
        einsums_arg = False
        for arg in sys.argv[1:]:
            if einsums_arg:
                pass_args.append(arg)
                einsums_arg = False
            elif arg == "--einsums":
                einsums_arg = True

    core.initialize(pass_args)


initialize()

atexit.register(core.finalize)
