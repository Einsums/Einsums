#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ProfileHeaders
    Einsums/Profile/Profile.hpp Einsums/Profile/LabeledSection.hpp Einsums/Profile/Section.hpp
    Einsums/Profile/Timer.hpp
    Einsums/Profile/Detail/PerformanceCounter.hpp
)

set(ProfileSources Section.cpp Timer.cpp)

if(APPLE)
  list(APPEND ProfileSources macOS.cpp)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
endif()

if(EINSUMS_WITH_OMP_TOOLS_H)
  list(APPEND ProfileSource OMPT.cpp)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums Profile
  SOURCES ${ProfileSources}
  HEADERS ${ProfileHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_Logging Einsums_Print Einsums_StringUtil
  CMAKE_SUBDIRS examples tests
)
