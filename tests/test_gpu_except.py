# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import pytest
import einsums as ein

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "These are GPU tests. Can't test them without a GPU.")
def test_throw_hip() :
    with pytest.raises(ein.core.errors.ErrorInvalidValue) :
        ein.core.throw_hip(1)
