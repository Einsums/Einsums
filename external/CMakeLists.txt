#---------------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#---------------------------------------------------------------------------------------------------

if (NOT WIN32)
    set(USE_TBLIS ON)
    include(ExternalProject)
    #set(TBLIS_CONFIG_FLAGS "--prefix=${CMAKE_INSTALL_PREFIX}")
    if (NOT BUILD_SHARED_LIBS)
        set(TBLIS_CONFIG_FLAGS "--enable-shared=no ${TBLIS_CONFIG_FLAGS}")
    endif()
    if (OpenMP_Found)
        set(TBLIS_CONFIG_FLAGS "--enable-thread-model=openmp ${TBLIS_CONFIG_FLAGS}")
    endif()

    if (NOT "$ENV{LIBS}" MATCHES "-lstdc\\+\\+")
        set(ENV{LIBS} "$ENV{LIBS} -lstdc++")
    endif()

    message(STATUS "${cyan} Building TBLIS with flags (allow three minutes): ${reset}${TBLIS_CONFIG_FLAGS}")

    ExternalProject_Add(tblis
        SOURCE_DIR ${Einsums_SOURCE_DIR}/external/tblis
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tblis-build
        LOG_CONFIGURE TRUE
        USES_TERMINAL_CONFIGURE TRUE
        CONFIGURE_COMMAND ${Einsums_SOURCE_DIR}/external/tblis/configure --prefix=<INSTALL_DIR> ${TBLIS_CONFIG_FLAGS}
        LOG_BUILD TRUE
        USES_TERMINAL_BUILD TRUE
        BUILD_COMMAND make
        LOG_INSTALL TRUE
        USES_TERMINAL_INSTALL TRUE
        INSTALL_COMMAND make install
    )
endif()