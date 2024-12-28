//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <functional>
#include <string>

#include "ShutdownFunction.hpp"
#include "StartupFunction.hpp"

#if defined(EINSUMS_APPLICATION_NAME_DEFAULT) && !defined(EINSUMS_APPLICATION_NAME)
#    define EINSUMS_APPLICATION_NAME EINSUMS_APPLICATION_NAME_DEFAULT
#endif

#if !defined(EINSUMS_APPLICATION_STRING)
#    if defined(EINSUMS_APPLICAITON_NAME)
#        define EINSUMS_APPLICATION_STRING EINSUMS_PP_STRING(EINSUMS_APPLICATION_NAME)
#    else
#        define EINSUMS_APPLICATION_STRING "unknown Einsums application"
#    endif
#endif

namespace einsums {

namespace detail {
// Default params to initialize with
[[maybe_unused]] static int    dummy_argc      = 1;
[[maybe_unused]] static char   app_name[]      = EINSUMS_APPLICATION_STRING;
static char                   *default_argv[2] = {app_name, nullptr};
[[maybe_unused]] static char **dummy_argv      = default_argv;
} // namespace detail

struct InitParams {
    mutable StartupFunctionType  startup;
    mutable ShutdownFunctionType shutdown;
};

EINSUMS_EXPORT int init(std::function<int(ConfigMap<std::string> &)> f, int argc, char const *const *argv,
                        InitParams const &params = InitParams());
EINSUMS_EXPORT int init(std::function<int(int, char **)> f, int argc, char const *const *argv, InitParams const &params = InitParams());
EINSUMS_EXPORT int init(std::function<int()> f, int argc, char const *const *argv, InitParams const &params = InitParams());
EINSUMS_EXPORT int init(std::nullptr_t, int argc, char const *const *argv, InitParams const &params = InitParams());

} // namespace einsums