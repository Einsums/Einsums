#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(APPLE)
    list(APPEND CMAKE_INSTALL_RPATH "\@loader_path")
elseif(NOT MSVC)
    list(APPEND CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
