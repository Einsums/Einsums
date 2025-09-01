#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_AddDefinitions)

if(NOT TARGET einsums_dependencies_allocator)

  if(NOT EINSUMS_WITH_MALLOC)
    set(EINSUMS_WITH_MALLOC
        CACHE STRING "Use the specific allocator. Supported allocators are mimalloc and system."
              ${DEFAULT_MALLOC}
    )
    set(allocator_error
        "The default allocator for your system is ${DEFAULT_MALLOC}, but ${DEFAULT_MALLOC} could not be found. "
        "The system allocator has poor performance. As such ${DEFAULT_MALLOC} is a strong optional requirement. "
        "Being aware of the performance hit, you can override this default and get rid of this dependency by setting -DEINSUMS_WITH_MALLOC=system. "
        "Valid options for EINSUMS_WITH_MALLOC are: system and mimalloc."
    )
  else()
    set(allocator_error
        "EINSUMS_WITH_MALLOC was set to ${EINSUMS_WITH_MALLOC}, but ${EINSUMS_WITH_MALLOC} could not be found. "
        "Valid options for EINSUMS_WITH_MALLOC are: system and mimalloc."
    )
  endif()

  string(TOUPPER "${EINSUMS_WITH_MALLOC}" EINSUMS_WITH_MALLOC_UPPER)

  add_library(einsums_dependencies_allocator INTERFACE IMPORTED)

  if(NOT EINSUMS_WITH_MALLOC_UPPER STREQUAL "SYSTEM")

    # ##############################################################################################
    # MIMALLOC
    if("${EINSUMS_WITH_MALLOC_UPPER}" STREQUAL "MIMALLOC")

      find_package(mimalloc)
      if(NOT mimalloc_FOUND)
        einsums_error(${allocator_error})
      endif()
      target_link_libraries(einsums_dependencies_allocator INTERFACE mimalloc)
      set(EINSUMS_MALLOC_LIBRARY mimalloc)
      if(MSVC)
        target_compile_options(einsums_dependencies_allocator INTERFACE /INCLUDE:mi_version)
      endif()

      einsums_warn(
        "einsums is using mimalloc as the allocator. Typically, exporting the following environment variables will further improve performance: MIMALLOC_EAGER_COMMIT_DELAY=0 and MIMALLOC_ALLOW_LARGE_OS_PAGES=1."
      )
    endif()
  endif()

  if("${EINSUMS_WITH_MALLOC_UPPER}" MATCHES "SYSTEM")
    if(NOT MSVC)
      einsums_warn("einsums will perform poorly without mimalloc. See docs for more info,")
    endif()
  endif()

  einsums_info("Using ${EINSUMS_WITH_MALLOC} allocator.")

  # convey selected allocator type to the build configuration
  if(NOT EINSUMS_FIND_PACKAGE)
    einsums_add_config_define(EINSUMS_HAVE_MALLOC "\"${EINSUMS_WITH_MALLOC}\"")
    einsums_add_config_define(EINSUMS_HAVE_MALLOC_${EINSUMS_WITH_MALLOC_UPPER})
  endif()
endif()
