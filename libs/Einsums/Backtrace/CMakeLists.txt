#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(BacktraceHeaders)

set(BacktraceSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Backtrace
  GLOBAL_HEADER_GEN ON
  SOURCES ${BacktraceSources}
  HEADERS ${BacktraceHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
