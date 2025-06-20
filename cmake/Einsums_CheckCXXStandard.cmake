#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# We require at least C++20. However, if a higher standard is set by the user in CMAKE_CXX_STANDARD
# that requirement has to be propagated to users of einsums as well. Ideally, users should not set
# CMAKE_CXX_STANDARD when building einsums.
einsums_option(
  EINSUMS_WITH_CXX_STANDARD STRING "C++ standard to use for compiling einsums (default: 20)" "20"
  ADVANCED CATEGORY "Generic"
)

if(EINSUMS_WITH_CXX_STANDARD LESS 20)
  einsums_error(
    "You've set EINSUMS_WITH_CXX_STANDARD to ${EINSUMS_WITH_CXX_STANDARD}, which is less than 20 which is the minimum required by einsums"
  )
endif()

if(DEFINED CMAKE_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD STREQUAL EINSUMS_WITH_CXX_STANDARD)
  einsums_error(
    "You've set CMAKE_CXX_STANDARD to ${CMAKE_CXX_STANDARD} and EINSUMS_WITH_CXX_STANDARD to ${EINSUMS_WITH_CXX_STANDARD}. Please unset CMAKE_CXX_STANDARD."
  )
endif()

set(CMAKE_CXX_STANDARD ${EINSUMS_WITH_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# We explicitly set the default to 98 to force CMake to emit a -std=c++XX flag. Some compilers
# (clang) have a different default standard for cpp and cu files, but CMake does not know about this
# difference. If the standard is set to the .cpp default in CMake, CMake will omit the flag,
# resulting in the wrong standard for .cu files.
set(CMAKE_CXX_STANDARD_DEFAULT 98)

einsums_info("Using C++${EINSUMS_WITH_CXX_STANDARD}")
