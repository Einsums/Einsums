#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import sys

import pytest

if sys.version_info < (3, 8):
    from importlib_metadata import version
else:
    from importlib.metadata import version

sphinx_vesion = tuple(int(d) for d in version("sphinx").split(".")[:2])

pytest_plugins = "sphinx.testing.fixtures"

if sphinx_vesion < (7, 3):
    from sphinx.testing.path import path


    @pytest.fixture(scope="session")
    def rootdir():
        return path(__file__).parent.abspath() / "roots"
else:
    from pathlib import Path


    @pytest.fixture(scope="session")
    def rootdir():
        return Path(__file__).parent.absolute() / "roots"
