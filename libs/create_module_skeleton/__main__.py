import os
import sys
import argparse

import build_structure, configure_toplevel

def main() :

    parser = argparse.ArgumentParser(prog = "create_module_skeleton", description = "Creates a module skeleton for developing in Einsums.")

    parser.add_argument("library", help = "The name of the top-level library.")
    parser.add_argument("module", help = "The name of the module within the top-level library.")
    parser.add_argument("--python", help = "Set this flag if the module is a Python extension module.", action= "store_true")
    parser.add_argument("--gpu", help = "Set this flag if the module is a GPU module.", action = "store_true")
    parser.add_argument("--python-name", help = "The name of the top-level Python module.")

    args = parser.parse_args()

    module = args.module
    library = args.library
    lib_symb = library if not args.python else "${EINSUMS_PYTHON_LIB_NAME}"
    if args.python_name is None :
        args.python_name = library

    build_structure.build_structure(os.curdir, library, module, lib_symb = lib_symb, **vars(args))

    configure_toplevel.configure_python(os.curdir, library, **vars(args))
    configure_toplevel.configure_cmake(os.curdir, library, lib_symb = lib_symb, **vars(args))
    configure_toplevel.configure_module_docs(os.curdir, library, lib_symb = lib_symb, **vars(args))

if __name__ == "__main__" :
    main()
