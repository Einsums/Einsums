//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

namespace einsums {

namespace detail {
struct EINSUMS_EXPORT Log {
    int         level;
    std::string destination;
    std::string format;
};

struct EINSUMS_EXPORT System {
    pid_t       pid{-1};
    std::string executable_prefix;
};

struct EINSUMS_EXPORT Einsums {
    std::string master_yaml_path;

    bool install_signal_handlers{true};
    bool attach_debugger{true};
    bool diagnostics_on_terminate{true};

    Log log;
};
} // namespace detail

struct EINSUMS_EXPORT RuntimeConfiguration {
    detail::System  system;
    detail::Einsums einsums;

    RuntimeConfiguration();

  private:
    void pre_initialize();
};

} // namespace einsums