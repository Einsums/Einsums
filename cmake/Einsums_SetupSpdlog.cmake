#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(SPDLOG_INSTALL ON)
set(SPDLOG_FMT_EXTERNAL ON)
fetchcontent_declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.x
  FIND_PACKAGE_ARGS
  1
)
fetchcontent_makeavailable(spdlog)
