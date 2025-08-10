#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# For the moment always use Tracy
fetchcontent_declare(
  tracy
  GIT_REPOSITORY https://github.com/wolfpld/tracy.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
fetchcontent_makeavailable(tracy)
einsums_add_config_define_namespace(DEFINE EINSUMS_HAVE_TRACY NAMESPACE PROFILE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ProfileHeaders Einsums/Profile/Profile.hpp)

set(ProfileSources Profile.cpp)

if(EINSUMS_WITH_OMP_TOOLS_H)
  list(APPEND ProfileSource OMPT.cpp)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Profile
  SOURCES ${ProfileSources}
  HEADERS ${ProfileHeaders}
  DEPENDENCIES Tracy::TracyClient
  MODULE_DEPENDENCIES Einsums_Config Einsums_Logging Einsums_Print Einsums_Preprocessor
  CMAKE_SUBDIRS examples tests
)
