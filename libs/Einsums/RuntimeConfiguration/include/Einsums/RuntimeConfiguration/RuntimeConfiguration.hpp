//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#endif

#include <argparse/argparse.hpp>
#include <memory>

namespace einsums {

namespace detail {

/**
 * Settings for the logger.
 */
struct EINSUMS_EXPORT Log {
    /**
     * The log level. This is compatible with spdlog::level::level_enum.
     * Default value is 2, currently.
     */
    int level;
    /**
     * The destination sink for the logging messages.
     * Default value is "cerr" which is mapped to std::cerr.
     */
    std::string destination;
    /**
     * The format string for logging messages.
     */
    std::string format;
};

/**
 * Information regarding the running executable. These are filled in automatically
 * by the RuntimeConfiguration constructor.
 */
struct EINSUMS_EXPORT System {
    /// The process id of the running instance.
    int pid{-1};
    /**
     * The root directory of the executable.
     * For example, if the executable is /usr/local/bin/einsums then the executable prefix is /usr/local.
     */
    std::string executable_prefix;
};

struct EINSUMS_EXPORT Einsums {
    /**
     * This will eventually be the master config file name.
     *
     * Defaults are set in RuntimeConfiguration::pre_initialize() then
     * The master_yaml_path will be read in changing the defaults.
     */
    std::string master_yaml_path;

    /**
     * Install Einsums signal handlers.
     *
     * Can be useful in debugging segfaults, bus errors, etc.
     */
    bool install_signal_handlers{true};

    /**
     * Provide a mechanism to attach a debugger to the running instance.
     *
     * For example, if install_signal_handlers is true and attach_debugger is true
     * then when a signal is caught the user will be presented with a message
     * telling them how to attach a debugger to the instance.
     */
    bool attach_debugger{true};

    /**
     * Provide more detailed information when a signal is handled, an
     * unhandled exception is caught, or an assertion is thrown.
     *
     * Diagnostics include version and compilation configuration
     * and a stack trace of what was caught.
     */
    bool diagnostics_on_terminate{true};

    /// Settings for the logging submodule.
    Log log;
};
} // namespace detail

/**
 * Handles the current configuration state of the running instance.
 *
 * Currently, defaults are handled in pre_initialize. Eventually,
 * some kind of configuration file will be implemented that will
 * override the defaults set in pre_initialize.
 *
 * The current instance of the RuntimeConfiguration can be obtained
 * from Runtime::config() or from runtime_config() functions.
 */
struct EINSUMS_EXPORT RuntimeConfiguration {
    detail::System  system;
    detail::Einsums einsums;

    struct {
        int                argc;
        char const *const *argv;
    } original;

    std::unique_ptr<argparse::ArgumentParser> argument_parser;

    /**
     * Constructor of the runtime configuration object of einsums.
     *
     * @param argc the argc argument from main
     * @param argv the argv argument from main
     */
    RuntimeConfiguration(int argc, char const *const *argv, std::function<void(argparse::ArgumentParser &)> const &user_command_line = {});
    RuntimeConfiguration() = delete;

  private:
    /**
     * Currently sets reasonable defaults for the development of Einsums.
     */
    void pre_initialize();

    /**
     * Parse the command line arguments provided in argc and argv.
     */
    void parse_command_line(std::function<void(argparse::ArgumentParser &)> const &user_command_line = {});
};

} // namespace einsums