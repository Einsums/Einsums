#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_PROFILER)
  set(EINSUMS_WITH_TRACY_DEFAULT OFF)
  einsums_option(
    EINSUMS_WITH_TRACY BOOL "Enable support for Tracy (default: ${EINSUMS_WITH_TRACY_DEFAULT}"
    ${EINSUMS_WITH_TRACY_DEFAULT} ADVANCED CATEGORY "Build Targets"
  )

  if(EINSUMS_WITH_TRACY)
    include(FetchContent)
    # For the moment always use Tracy
    fetchcontent_declare(
      tracy
      GIT_REPOSITORY https://github.com/wolfpld/tracy.git
      GIT_TAG master
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
    )
    fetchcontent_makeavailable(tracy)
    einsums_add_config_define(EINSUMS_HAVE_TRACY)
  endif()
endif()
