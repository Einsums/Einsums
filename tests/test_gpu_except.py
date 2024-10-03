import pytest
import einsums as ein

@pytest.mark.skipif(not ein.core.gpu_enabled(), reason = "These are GPU tests. Can't test them without a GPU.")
def test_throw_hip() :
    with pytest.raises(ein.gpu_except.ErrorInvalidValue) :
        ein.core.throw_hip(1)
