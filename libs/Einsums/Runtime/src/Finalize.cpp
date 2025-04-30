//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <cstdlib>

namespace einsums {

namespace detail {

static std::list<std::function<void()>> __deleters{};

void register_free_pointer(std::function<void()> f) {
    __deleters.push_back(f);
}

} // namespace detail

int finalize() {
    auto &rt = runtime();
    rt.call_shutdown_functions(true);
    EINSUMS_LOG_INFO("ran pre-shutdown functions");
    rt.call_shutdown_functions(false);
    EINSUMS_LOG_INFO("ran shutdown functions");

    auto &global_config = GlobalConfigMap::get_singleton();

    if (global_config.get_bool("profiler-report")) {
        auto filename = global_config.get_string("profiler-filename");
        if (filename == "stdout") {
            profile::detail::Profiler::get().format_results(std::cout);
        } else if (filename == "stderr") {
            profile::detail::Profiler::get().format_results(std::cerr);
        } else {
            std::ofstream ofs(filename, global_config.get_bool("profiler-append") ? std::ios::out | std::ios::app : std::ios::out);
            profile::detail::Profiler::get().format_results(ofs);
        }
    }

    // this function destroys the runtime.
    rt.deinit_global_data();

    // This is the only explicit finalization routine. This is because the runtime depends on the
    // profiler. If the profiler used the normal finalization, then it would also depend on the runtime.
    // This would cause a dependency error.
    // profile::finalize();

    // Free lost pointers.
    for (auto fn : detail::__deleters) {
        fn();
    }

    return EXIT_SUCCESS;
}

} // namespace einsums