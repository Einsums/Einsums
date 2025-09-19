"""Version tests."""

from sphinxcontrib.moderncmakedomain import __version__


def test_version():
    """Test that version has at least 3 parts."""
    assert __version__.count(".") >= 2
