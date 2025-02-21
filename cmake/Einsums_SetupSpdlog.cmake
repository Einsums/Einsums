#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(SPDLOG_INSTALL TRUE)
set(SPDLOG_FMT_EXTERNAL TRUE)
fetchcontent_declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz
  FIND_PACKAGE_ARGS 1.15
)

# Ensure the option is set before spdlog is configured
set(SPDLOG_MSVC_UTF8 FALSE CACHE BOOL "Disable MSVC UTF-8 support in spdlog" FORCE)

fetchcontent_makeavailable(spdlog)
