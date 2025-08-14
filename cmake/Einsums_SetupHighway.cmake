#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(HWY_ENABLE_TESTS
    OFF
    CACHE BOOL "Enable HWY tests"
)

# Grab Highway from GitHub
fetchcontent_declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG a0b07d08a6a0b1f298cbb02a91f3dedec02e1633 # this is master as of 8/13/2025
  EXCLUDE_FROM_ALL
)

# Fetch and make targets available
fetchcontent_makeavailable(highway)
