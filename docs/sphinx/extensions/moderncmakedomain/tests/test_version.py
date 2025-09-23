# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Version tests."""

from sphinxcontrib.moderncmakedomain import __version__


def test_version():
    """Test that version has at least 3 parts."""
    assert __version__.count(".") >= 2
