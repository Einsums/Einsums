# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest # pylint: disable=unused-import
import sys
import os
import re

def test_find_core():
    einsums_is_findable = False
    for dir in sys.path :
        if not os.path.isdir(dir + os.path.sep + "einsums") :
            continue
        files = os.listdir(dir + os.path.sep + "einsums")
        
        for file in files :
            if re.match("core\\.[_a-zA-Z0-9-]+\\.(pyd|so)", file) :
                print("Found " + dir + os.path.sep + "einsums" + os.path.sep + file)
                einsums_is_findable = True
    assert einsums_is_findable

def test_import() :
    assert True
    print("Importing einsums.", flush = True)
    try:
        import einsums as ein # pylint: disable=import-outside-toplevel,unused-import
    except (ModuleNotFoundError, ImportError) :
        import pyeinsums as ein
    print("Imported einsums.", flush=True)

    assert True
    
if __name__ == "__main__":
    test_find_core()
    test_import()