//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

namespace einsums {

namespace detail {
struct System {
    pid_t       pid{-1};
    std::string executable_prefix;
};

struct Einsums {
    std::string master_yaml_path;
};
} // namespace detail

struct RuntimeConfiguration {
    detail::System system;
};

} // namespace einsums