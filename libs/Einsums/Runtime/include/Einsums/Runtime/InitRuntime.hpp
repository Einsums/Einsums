//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Runtime/StartupFunction.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

#include <functional>
#include <string>

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

EINSUMS_EXPORT int initialize(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params = InitParams());
EINSUMS_EXPORT int initialize(std::function<int()> f, int argc, char **argv, InitParams const &params = InitParams());
EINSUMS_EXPORT int initialize(std::nullptr_t, int argc, char **argv, InitParams const &params = InitParams());

/// \brief Start the runtime.
///
/// \param f entry point of the first task on the einsums runtime. f will be passed all non-einsums
/// command line arguments.
/// \param argc number of arguments in argv
/// \param argv array of arguments. The first element is ignored.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is stopped
/// \post the runtime is running
EINSUMS_EXPORT void start(std::function<int(int, char **)> f, int argc, char const *const *argv, InitParams const &params = InitParams());

/// \brief Start the runtime.
///
/// \param f entry point of the first task on the einsums runtime
/// \param argc number of arguments in argv
/// \param argv array of arguments. The first element is ignored.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not running
EINSUMS_EXPORT void start(std::function<int()> f, int argc, char const *const *argv, InitParams const &params = InitParams());

EINSUMS_EXPORT void start(std::nullptr_t, int argc, char const *const *argv, InitParams const &params = InitParams());

/// \brief Start the runtime.
///
/// No task is created on the runtime.
///
/// \param argc number of arguments in argv
/// \param argv array of arguments. The first element is ignored.
///
/// \pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
/// \pre the runtime is not initialized
/// \post the runtime is running
EINSUMS_EXPORT void start(int argc, char const *const *argv, InitParams const &params = InitParams());

/// \brief Signal the runtime that it may be stopped.
///
/// Until \ref einsums::finalize() has been called, \ref einsums::stop() will not return. This
/// function exists to distinguish between the runtime being idle but still expecting work to be
/// scheduled on it and the runtime being idle and ready to be shutdown. Unlike \ref
/// einsums::stop(), \ref einsums::finalize() can be called from within or outside the runtime.
///
/// \pre the runtime is initialized
EINSUMS_EXPORT void finalize();

} // namespace einsums