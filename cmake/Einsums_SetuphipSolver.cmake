#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

# fetchcontent_declare(hipsolver GIT_REPOSITORY https://github.com/ROCm/hipSOLVER.git
# FIND_PACKAGE_ARGS)

# fetchcontent_makeavailable(hipsolver)

find_package(hipsolver REQUIRED)
