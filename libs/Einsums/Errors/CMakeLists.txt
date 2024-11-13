#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ErrorsHeaders
    Einsums/Errors/Error.hpp
    Einsums/Errors/ErrorCode.hpp
    Einsums/Errors/Exception.hpp
    Einsums/Errors/ExceptionForward.hpp
    Einsums/Errors/ExceptionInfo.hpp
    Einsums/Errors/ThrowException.hpp
    Einsums/Errors/TryCatchExceptionPointer.hpp
)

set(ErrorsSources ErrorCode.cpp Exception.cpp ThrowException.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Errors
  GLOBAL_HEADER_GEN ON
  SOURCES ${ErrorsSources}
  HEADERS ${ErrorsHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Assertion Einsums_Config Einsums_Preprocessor
  CMAKE_SUBDIRS examples tests
)
