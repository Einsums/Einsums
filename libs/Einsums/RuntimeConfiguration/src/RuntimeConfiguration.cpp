//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ThrowException.hpp>
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
    return p.parent_path().parent_path().string();
}
} // namespace detail

void RuntimeConfiguration::pre_initialize() {
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
    system.pid               = getpid();
    system.executable_prefix = detail::get_executable_prefix();

    einsums.master_yaml_path         = detail::get_executable_prefix();
    einsums.install_signal_handlers  = true;
    einsums.attach_debugger          = true;
    einsums.diagnostics_on_terminate = true;

    einsums.log.level       = 2;
    einsums.log.destination = "cerr";
    einsums.log.format      = "[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] [host:%j] [pid:%P] [tid:%t] [%s:%#/%!] %v";
}

RuntimeConfiguration::RuntimeConfiguration(int argc, char const *const argv[]) : argc(argc), argv(argv) {
    pre_initialize();
}

} // namespace einsums