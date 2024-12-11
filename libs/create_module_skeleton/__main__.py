import os
import sys
import argparse

import build_structure, configure_toplevel

def main() :

    parser = argparse.ArgumentParser(prog = "create_module_skeleton", description = "Creates a module skeleton for developing in Einsums.")

    parser.add_argument("library", help = "The name of the top-level library.")
    parser.add_argument("module", help = "The name of the module within the top-level library.")

    args = parser.parse_args()

    module = args.module
    library = args.library

    build_structure.build_structure(os.curdir, library, module, **vars(args))

    configure_toplevel.configure_cmake(os.curdir, library, **vars(args))
    configure_toplevel.configure_module_docs(os.curdir, library, **vars(args))

if __name__ == "__main__" :
    main()
