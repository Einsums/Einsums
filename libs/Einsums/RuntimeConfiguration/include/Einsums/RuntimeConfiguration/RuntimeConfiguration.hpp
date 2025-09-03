//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/TypeSupport/Observable.hpp>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#endif

#include <memory>

namespace einsums {

struct RuntimeOptions {

  protected:
    /// Install signal handlers
    bool install_signal_handlers = true;

    /// Provide a mechanism to attach debugger on detected errors
    bool attach_debugger = true;

    /// Print additional diagnostic information on termination
    bool diagnostics_on_terminate = true;

    struct { /// Set log level
#if defined(EINSUMS_DEBUG)
        int64_t level = SPDLOG_LEVEL_DEBUG;
#else
        int64_t level = SPDLOG_LEVEL_INFO;
#endif

        /// Set log destination
        std::string destination = "cerr";

        /// Log format
        std::string format = "[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%-8l%$] [%s:%#/%!] %v";
    } log;

    struct {
        /// Profile report?
        bool report = true;

        /// Profile filename
        std::string filename = "profile.txt";

        /// Append to profile file
        bool append = true;
    } profile;
};

/**
 * @brief Add a function to the list of startup functions to add module-specific command line arguments.
 */
EINSUMS_EXPORT void register_arguments(std::function<void()>);

/**
 * @struct RuntimeConfiguration
 *
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
    /**
     * @property original
     *
     * @todo Document.
     */
    std::vector<std::string> original;

    /**
     * Constructor of the runtime configuration object of einsums.
     *
     * @param argc the argc argument from main
     * @param argv the argv argument from main
     * @param user_command_line callback function that can be used to register additional command-line options
     */
    RuntimeConfiguration(int argc, char const *const *argv, std::function<void()> const &user_command_line = {});

    /**
     * Constructor of the runtime configuration object of einsums. This is used when argv has been packaged into a vector.
     *
     * @param argv The argv that has been packaged up.
     * @param user_command_line callback function that can be used to register additional command-line options
     */
    explicit RuntimeConfiguration(std::vector<std::string> const &argv, std::function<void()> const &user_command_line = {});

    RuntimeConfiguration() = delete;

  private:
    /**
     * Currently sets reasonable defaults for the development of Einsums.
     */
    void pre_initialize();

    /**
     * Parse the command line arguments provided in argc and argv. Returns unknown command line arguments.
     */
    std::vector<std::string> parse_command_line(std::function<void()> const &user_command_line = {});
};

} // namespace einsums