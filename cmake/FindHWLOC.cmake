#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(NOT TARGET einsums_internal::hwloc)
    find_package(PkgConfig QUIET)
    pkg_check_modules(PC_HWLOC QUIET hwloc)

    find_path(
            HWLOC_INCLUDE_DIR hwloc.h
            HINTS ${HWLOC_ROOT}
            ENV
            HWLOC_ROOT
            ${EINSUMS_HWLOC_ROOT}
            ${PC_HWLOC_MINIMAL_INCLUDEDIR}
            ${PC_HWLOC_MINIMAL_INCLUDE_DIRS}
            ${PC_HWLOC_INCLUDEDIR}
            ${PC_HWLOC_INCLUDE_DIRS}
            PATH_SUFFIXES include
    )

    find_library(
            HWLOC_LIBRARY
            NAMES hwloc libhwloc
            HINTS ${HWLOC_ROOT}
            ENV
            HWLOC_ROOT
            ${EINSUMS_HWLOC_ROOT}
            ${PC_HWLOC_MINIMAL_LIBDIR}
            ${PC_HWLOC_MINIMAL_LIBRARY_DIRS}
            ${PC_HWLOC_LIBDIR}
            ${PC_HWLOC_LIBRARY_DIRS}
            PATH_SUFFIXES lib lib64
    )

    # Set HWLOC_ROOT in case the other hints are used
    if (HWLOC_ROOT)
        # The call to file is for compatibility with windows paths
        file(TO_CMAKE_PATH ${HWLOC_ROOT} HWLOC_ROOT)
    elseif ("$ENV{HWLOC_ROOT}")
        file(TO_CMAKE_PATH $ENV{HWLOC_ROOT} HWLOC_ROOT)
    else ()
        file(TO_CMAKE_PATH "${HWLOC_INCLUDE_DIR}" HWLOC_INCLUDE_DIR)
        string(REPLACE "/include" "" HWLOC_ROOT "${HWLOC_INCLUDE_DIR}")
    endif ()

    set(HWLOC_LIBRARIES ${HWLOC_LIBRARY})
    set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})

    find_package_handle_standard_args(HWLOC DEFAULT_MSG HWLOC_LIBRARY HWLOC_INCLUDE_DIR)

    get_property(
            _type
            CACHE HWLOC_ROOT
            PROPERTY TYPE
    )
    if (_type)
        set_property(CACHE HWLOC_ROOT PROPERTY ADVANCED 1)
        if ("x${_type}" STREQUAL "xUNINITIALIZED")
            set_property(CACHE HWLOC_ROOT PROPERTY TYPE PATH)
        endif ()
    endif ()

    add_library(einsums_internal::hwloc INTERFACE IMPORTED)
    target_include_directories(einsums_internal::hwloc SYSTEM INTERFACE ${HWLOC_INCLUDE_DIR})
    target_link_libraries(einsums_internal::hwloc INTERFACE ${HWLOC_LIBRARIES})

    mark_as_advanced(HWLOC_ROOT HWLOC_LIBRARY HWLOC_INCLUDE_DIR)
endif()
