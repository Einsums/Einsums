#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}/cmake")

set({module_name}Headers)

set({module_name}Sources)

include(Einsums_AddModule)
einsums_add_module(
  {lib_symb} {module_name}
  GLOBAL_HEADER_GEN ON
  SOURCES ${{{module_name}Sources}}
  HEADERS ${{{module_name}Headers}}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
