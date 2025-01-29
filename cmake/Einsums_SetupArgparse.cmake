#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

fetchcontent_declare(
  argparse
  GIT_REPOSITORY https://github.com/Einsums/argparse.git
  FIND_PACKAGE_ARGS 3
)
fetchcontent_makeavailable(argparse)
