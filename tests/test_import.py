import pytest # pylint: disable=unused-import

def test_import() :
    print("Importing einsums.", flush = True)
    import einsums as ein # pylint: disable=import-outside-toplevel,unused-import
    print("Imported einsums.", flush=True)

if __name__ == "__main__" :
    print("Importing einsums.", flush = True)
    import einsums as ein # pylint: disable=import-outside-toplevel,unused-import
    print("Imported einsums.", flush=True)
