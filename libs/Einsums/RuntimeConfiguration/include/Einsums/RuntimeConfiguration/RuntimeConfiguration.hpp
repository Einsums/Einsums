//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TypeSupport/Observable.hpp>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#endif

#include <argparse/argparse.hpp>
#include <memory>

namespace einsums {

namespace detail {} // namespace detail

/**
 * @brief Creates a command line argument that can appear as a switch.
 *
 * Say you want to create a command line switch that looks like @c --dummy-flag , but
 * you also want a switch like @c --no-dummy-flag . When the first is present, the result will
 * be true, and when the second, the result will be false. This function sets up such a flag.
 *
 * @param arg The argument to add this behavior to.
 * @param output A reference to where the flag should be stored.
 * @param true_flag The name for the true flag. If this is present on the command line, @c true will be stored into the output.
 * @param false_flag The name for the false flag. If this is present on the command line, @c false will be stored into the output.
 * @param true_help Help statement for the true flag.
 * @param false_help Help statement for the false flag.
 * @param default_value The default value for the flag.
 */
EINSUMS_EXPORT void no_flag(argparse::ArgumentParser &arg, bool &output, std::string const &true_flag, std::string const &false_flag,
                            std::string const &true_help, std::string const &false_help, bool default_value);

/**
 * @copybrief no_flag(argparse::ArgumentParser &,bool &, std::string const &,std::string const &,std::string const &,std::string const &,bool)
 * @copydetail no_flag(argparse::ArgumentParser &,bool &, std::string const &,std::string const &,std::string const &,std::string const &,bool)
 *
 * @param arg The argument to add this behavior to.
 * @param output A reference to where the flag should be stored.
 * @param true_flag The name for the true flag. If this is present on the command line, @c true will be stored into the output.
 * @param false_flag The name for the false flag. If this is present on the command line, @c false will be stored into the output.
 * @param non_default_help Help statement for the flag corresponding to the opposite of the default value. The other flag will have a simple help statement.
 * @param default_value The default value for the flag.
 */
EINSUMS_EXPORT void no_flag(argparse::ArgumentParser &arg, bool &output, std::string const &true_flag, std::string const &false_flag,
                            std::string const &non_default_help, bool default_value);

EINSUMS_EXPORT void register_arguments(std::function<void(argparse::ArgumentParser &)>);

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
     * @property argument_parser
     *
     * @brief Holds the parser used to parse the arguments passed to Einsums.
     */
    std::unique_ptr<argparse::ArgumentParser> argument_parser;

    /**
     * Constructor of the runtime configuration object of einsums.
     *
     * @param argc the argc argument from main
     * @param argv the argv argument from main
     * @param user_command_line callback function that can be used to register additional command-line options
     */
    RuntimeConfiguration(int argc, char const *const *argv, std::function<void(argparse::ArgumentParser &)> const &user_command_line = {});

    /**
     * Constructor of the runtime configuration object of einsums. This is used when argv has been packaged into a vector.
     *
     * @param argv The argv that has been packaged up.
     * @param user_command_line callback function that can be used to register additional command-line options
     */
    explicit RuntimeConfiguration(std::vector<std::string> const                        &argv,
                                  std::function<void(argparse::ArgumentParser &)> const &user_command_line = {});

    RuntimeConfiguration() = delete;

  private:
    /**
     * Currently sets reasonable defaults for the development of Einsums.
     */
    void pre_initialize();

    /**
     * Parse the command line arguments provided in argc and argv. Returns unknown command line arguments.
     */
    std::vector<std::string> parse_command_line(std::function<void(argparse::ArgumentParser &)> const &user_command_line = {});
};

} // namespace einsums