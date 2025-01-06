//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

#include <filesystem>
#include <string>
#include <vector>

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#elif defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#    include <windows.h>
#elif defined(__linux) || defined(linux) || defined(__linux__)
#    include <filesystem>
#elif __APPLE__
#    include <mach-o/dyld.h>
#endif

#include <argparse/argparse.hpp>

namespace einsums {
namespace detail {
std::string get_executable_filename() {
    std::string r;

#if defined(EINSUMS_WINDOWS)
    char exe_path[MAX_PATH + 1] = {'\0'};
    if (!GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path))) {
        EINSUMS_THROW_EXCEPTION(system_error, "unable to find executable filename");
    }
    r = exe_path;
#elif defined(__linux) || defined(linux) || defined(__linux__)
    r = std::filesystem::canonical("/proc/self/exe").string();
#elif defined(__APPLE__)
    char          exe_path[PATH_MAX + 1];
    std::uint32_t len = sizeof(exe_path) / sizeof(exe_path[0]);

    if (0 != _NSGetExecutablePath(exe_path, &len)) {
        EINSUMS_THROW_EXCEPTION(system_error, "unable to find executable filename");
    }
    exe_path[len - 1] = '\0';
    r                 = exe_path;
#else
#    error Unsupported platform
#endif

    return r;
}

std::string get_executable_prefix() {
    std::filesystem::path p(get_executable_filename());
    std::string           prefix = p.parent_path().parent_path().string();

    return prefix;
}
} // namespace detail

void RuntimeConfiguration::pre_initialize() {
    EINSUMS_LOG_INFO("Setting default configuration values");
    /*
     * This routine will eventually contain a "master" yaml template that
     * will include all the default settings for Einsums and its subsystems.
     *
     * Once a yaml or some other file settings format is decided on and brought
     * in that file will be used after initially using this one.
     */
    std::vector<std::string> lines = {
        // clang-format off
        "system:",
        "    pid: " + std::to_string(getpid()),
        "    executable_prefix: " + detail::get_executable_prefix(),
        "einsums:",
        "    master_yaml_file: ${system.executable_prefix}"
        "    "
        // clang-format on
    };

    // For now set the values to their default values.
    system.pid                       = getpid();
    system.executable_prefix         = detail::get_executable_prefix();
    einsums.master_yaml_path         = detail::get_executable_prefix();
    einsums.install_signal_handlers  = true;
    einsums.attach_debugger          = true;
    einsums.diagnostics_on_terminate = true;
    einsums.log.level                = 3;
    einsums.log.destination          = "cerr";
    // einsums.log.format               = "[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] [host:%j] [pid:%P] [tid:%t] [%s:%#/%!] %v";
    einsums.log.format = "[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%-8l%$] [%s:%#/%!] %v";

    einsums.profiler.generate_report = true;
    einsums.profiler.filename        = "profile.txt";
    einsums.profiler.append          = true;
}

RuntimeConfiguration::RuntimeConfiguration(int argc, char const *const argv[],
                                           std::function<void(argparse::ArgumentParser &)> const &user_command_line)
    : original{.argc = argc, .argv = argv} {
    pre_initialize();

    parse_command_line(user_command_line);
}

void RuntimeConfiguration::parse_command_line(std::function<void(argparse::ArgumentParser &)> const &user_command_line) {
    EINSUMS_LOG_INFO("Configuring command line parser and parsing user provided command line");

    // Imperative that pre_initialize is called first as it is responsible for setting
    // default values. This is done in the constructor.
    // There should be a mechanism that allows the user to change the program name.
    argument_parser.reset(new argparse::ArgumentParser("einsums"));

    argument_parser->add_argument("--einsums:install-signal-handlers")
        .default_value(einsums.install_signal_handlers)
        .help("install signal handlers")
        .store_into(einsums.install_signal_handlers);
    argument_parser->add_argument("--einsums:attach-debugger")
        .default_value(einsums.attach_debugger)
        .help("provides mechanism to attach debugger on detected errors")
        .store_into(einsums.attach_debugger);
    argument_parser->add_argument("--einsums:diagnostics-on-terminate")
        .default_value(einsums.diagnostics_on_terminate)
        .help("print additional diagnostic information on termination")
        .store_into(einsums.diagnostics_on_terminate);

    argument_parser->add_argument("--einsums:log-level")
        .default_value(einsums.log.level)
        .help("set log level")
        .choices(0, 1, 2, 3, 4)
        .store_into(einsums.log.level);
    argument_parser->add_argument("--einsums:log-destination")
        .default_value(einsums.log.destination)
        .help("set log destination")
        .choices("cerr", "cout")
        .store_into(einsums.log.destination);
    argument_parser->add_argument("--einsums:log-format").default_value(einsums.log.format).store_into(einsums.log.format);

    argument_parser->add_argument("--einsums:profiler-generate_report")
        .default_value(einsums.profiler.generate_report)
        .help("generate profiling report")
        .store_into(einsums.profiler.generate_report);
    argument_parser->add_argument("--einsums:profiler-filename")
        .default_value(einsums.profiler.filename)
        .help("filename of the profiling report")
        .store_into(einsums.profiler.filename);
    argument_parser->add_argument("--einsums:profiler-append")
        .default_value(einsums.profiler.append)
        .help("append to an existing file")
        .store_into(einsums.profiler.append);

    // Allow the user to inject their own command line options
    if (user_command_line) {
        EINSUMS_LOG_INFO("adding user command line options");
        user_command_line(*argument_parser);
    }

    try {
        argument_parser->parse_args(original.argc, original.argv);
    } catch (std::exception const &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << argument_parser;
        std::exit(1);
    }
}

} // namespace einsums