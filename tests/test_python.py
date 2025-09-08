# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
This module just makes sure that Python actually works and that any issues are caused by Einsums.
"""

import pytest  # pylint: disable=unused-import
import numpy as np  # pylint: disable=unused-import


def test_python():
    """
    Make sure Python works by just asserting true.
    """
    assert True
