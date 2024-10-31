import pytest # pylint: disable=unused-import

def test_import() :
    import einsums as ein # pylint: disable=import-outside-toplevel,unused-import

if __name__ == "__main__" :
    print("Importing einsums.")
    import einsums as ein
    print("Imported einsums.")
