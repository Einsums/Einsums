#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

fetchcontent_declare(
  librett
  GIT_REPOSITORY https://github.com/victor-anisimov/Librett.git
  FIND_PACKAGE_ARGS
)

fetchcontent_makeavailable(librett)

if(NOT TARGET librett::librett)
  message(FATAL_ERROR "Did not find librett!")
endif()
