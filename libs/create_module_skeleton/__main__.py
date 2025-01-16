#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import os
import sys
import argparse

import build_structure, configure_toplevel


def build_new(library, module, args):
    lib_symb = library if not args.python else "${EINSUMS_PYTHON_LIB_NAME}"
    if args.python_name is None:
        args.python_name = library

    build_structure.build_structure(
        os.curdir, library, module, lib_symb=lib_symb, **vars(args)
    )

    configure_toplevel.configure_python(os.curdir, library, **vars(args))
    configure_toplevel.configure_cmake(
        os.curdir, library, lib_symb=lib_symb, **vars(args)
    )
    configure_toplevel.configure_module_docs(
        os.curdir, library, lib_symb=lib_symb, **vars(args)
    )


def reindex(libraries=None):
    if libraries is None or len(libraries) == 0:
        libraries = filter(
            lambda x: os.path.isfile(os.path.join(os.curdir, x, "CMakeLists.txt")),
            os.listdir(),
        )

    for lib in libraries:
        lib_symb = lib
        if os.path.isfile(os.path.join(os.curdir, lib, ".is_python_lib")) :
            lib_symb = "${EINSUMS_PYTHON_LIB_NAME}"
            configure_toplevel.configure_python(os.curdir, lib, python = True, python_name = lib)

        configure_toplevel.configure_cmake(os.curdir, lib, lib_symb = lib_symb)
        configure_toplevel.configure_module_docs(os.curdir, lib, lib_symb = lib_symb)

def main():
    parser = argparse.ArgumentParser(
        prog="create_module_skeleton",
        description="Creates a module skeleton for developing in Einsums.",
        usage="""
        
%(prog)s --reindex [LIBRARIES, ...]
%(prog)s [OPTIONS] LIBRARY_NAME MODULE_NAME""",
    )

    parser.add_argument(
        "--python",
        help="Set this flag if the module is a Python extension module.",
        action="store_true",
    )
    parser.add_argument(
        "--gpu",
        help="Set this flag if the module is a GPU module.",
        action="store_true",
    )
    parser.add_argument(
        "--python-name", help="The name of the top-level Python module."
    )
    parser.add_argument(
        "--reindex",
        help="Reindex the libraries. The libraries may be specified afterwards. This is incompatible with other options.",
        action="store_true"
    )
    parser.add_argument(
        "--rebuild",
        help="Adds new files that were added to the template but do not exist in the output structure. It also re-indexes.",
        action="store_true"
    )

    known_args, unknown_args = parser.parse_known_intermixed_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if known_args.reindex:
        reindex(unknown_args)
    else:
        assert len(unknown_args) == 2
        build_new(unknown_args[0], unknown_args[1], known_args)


if __name__ == "__main__":
    main()
