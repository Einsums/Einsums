#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(AssertionHeaders Einsums/Assert.hpp Einsums/Assertion/EvaluateAssert.hpp)

set(AssertionSources Assertion.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Assertion
  SOURCES ${AssertionSources}
  HEADERS ${AssertionHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_Debugging Einsums_Preprocessor
  CMAKE_SUBDIRS examples tests
)
