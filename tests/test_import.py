import pytest # pylint: disable=unused-import

def test_import() :
    assert True
    print("Importing einsums.", flush = True)
    import einsums as ein # pylint: disable=import-outside-toplevel,unused-import
    print("Imported einsums.", flush=True)

    assert True