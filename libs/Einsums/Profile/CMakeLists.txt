#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ProfileHeaders Einsums/Profile/Profile.hpp)

set(ProfileSources Profile.cpp)

if(EINSUMS_WITH_OMP_TOOLS_H)
  list(APPEND ProfileSource OMPT.cpp)
endif()

set(ProfileDeps)
if(TARGET Tracy::TracyClient)
  list(APPEND ProfileDeps Tracy::TracyClient)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Profile
  SOURCES ${ProfileSources}
  HEADERS ${ProfileHeaders}
  DEPENDENCIES ${ProfileDeps}
  MODULE_DEPENDENCIES Einsums_Config Einsums_Logging Einsums_Print Einsums_Preprocessor
  CMAKE_SUBDIRS examples tests
)
