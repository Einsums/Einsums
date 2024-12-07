#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# find required packages
if (EINSUMS_WITH_DOCUMENTATION)
    find_package(Doxygen)
    find_package(Sphinx)
    #  find_package(Breathe)

    if (NOT Sphinx_FOUND)
        einsums_error(
                "Sphinx is unavailable, sphinx documentation generation disabled. Set Sphinx_ROOT to your sphinx-build installation directory."
        )
        set(EINSUMS_WITH_DOCUMENTATION OFF)
        #  elseif(NOT Breathe_FOUND)
        #    einsums_error(
        #      "Breathe is unavailable, sphinx documentation generation disabled. Set Breathe_APIDOC_ROOT to your breathe-apidoc installation directory."
        #    )
        #    set(EINSUMS_WITH_DOCUMENTATION OFF)
    elseif (NOT DOXYGEN_FOUND)
        einsums_error(
                "Doxygen tool is unavailable, sphinx documentation generation disabled. Add the doxygen executable to your path or set the DOXYGEN_EXECUTABLE variable manually."
        )
        set(EINSUMS_WITH_DOCUMENTATION OFF)
    endif ()
endif ()
