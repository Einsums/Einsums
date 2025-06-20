#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)
fetchcontent_declare(
  cpptrace
  GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
  GIT_TAG v0.7.3 # <HASH or TAG>
  FIND_PACKAGE_ARGS
  0.7
)
fetchcontent_makeavailable(cpptrace)
