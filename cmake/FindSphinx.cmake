#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# compatibility with older CMake versions
if (SPHINX_ROOT AND NOT Sphinx_ROOT)
    set(Sphinx_ROOT
            ${SPHINX_ROOT}
            CACHE PATH "Sphinx base directory"
    )
    unset(SPHINX_ROOT CACHE)
endif ()

find_program(
        Sphinx_EXECUTABLE
        NAMES sphinx-build sphinx-build2
        PATHS ${Sphinx_ROOT} ENV SPHINX_ROOT
        DOC "Path to sphinx-build executable"
)

if (Sphinx_EXECUTABLE)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Sphinx DEFAULT_MESSAGE Sphinx_EXECUTABLE)
endif ()
