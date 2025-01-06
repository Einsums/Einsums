#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

include(FetchContent)
fetchcontent_declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.13.6
  FIND_PACKAGE_ARGS 2.13
)

FetchContent_MakeAvailable(pybind11)