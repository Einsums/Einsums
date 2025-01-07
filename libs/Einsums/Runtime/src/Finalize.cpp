//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/Timer.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>

namespace einsums {
void finalize() {
    auto &rt = runtime();
    rt.call_shutdown_functions(true);
    EINSUMS_LOG_INFO("ran pre-shutdown functions");
    rt.call_shutdown_functions(false);
    EINSUMS_LOG_INFO("ran shutdown functions");

    detail::Profiler const &prof = rt.config().einsums.profiler;
    if (prof.generate_report) {
        profile::report(prof.filename, prof.append);
    }

    // this function destroys the runtime.
    rt.deinit_global_data();

    // This is the only explicit finalization routine. This is because the runtime depends on the 
    // profiler. If the profiler used the normal finalization, then it would also depend on the runtime.
    // This would cause a dependency error.
    profile::finalize();

    EINSUMS_LOG_INFO("einsums shutdown completed");

}
} // namespace einsums