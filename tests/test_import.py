# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest # pylint: disable=unused-import

def test_import() :
    assert True
    print("Importing einsums.", flush = True)
    import einsums as ein # pylint: disable=import-outside-toplevel,unused-import
    print("Imported einsums.", flush=True)

    assert True