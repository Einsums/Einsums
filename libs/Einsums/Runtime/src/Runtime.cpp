//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Debugging/AttachDebugger.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <csignal>

namespace einsums::detail {

std::list<StartupFunctionType>  global_pre_startup_functions;
std::list<StartupFunctionType>  global_startup_functions;
std::list<ShutdownFunctionType> global_pre_shutdown_functions;
std::list<ShutdownFunctionType> global_shutdown_functions;

[[noreturn]] EINSUMS_EXPORT void termination_handler(int signum) {
    if (signum != SIGINT /* && get_config_entry("einsums.attach_debugger", "") == "exception" */) {
        util::attach_debugger();
    }

    // TODO: If einsums.diagnostics_on_terminate is true then print out a lot of information.

    std::abort();
}

static bool exit_called = false;

void on_exit() noexcept {
    exit_called = true;
}

void on_abort(int) noexcept {
    exit_called = true;
    std::exit(-1);
}

Runtime::Runtime(RuntimeConfiguration &rtcfg, bool initialize) : _rtcfg(rtcfg) {
    if (initialize) {
        init();
    }
}

RuntimeState Runtime::state() const {
    return _state;
}

void Runtime::state(RuntimeState state) {
    // TODO: Log the state change.
    _state = state;
}

RuntimeConfiguration &Runtime::config() {
    return _rtcfg;
}

RuntimeConfiguration const &Runtime::config() const {
    return _rtcfg;
}

void Runtime::init() {
    try {
        // TODO: This would be a good place to create and initialize a thread pool

        // Copy over all startup functions registered so far.
        for (StartupFunctionType &f : global_pre_startup_functions) {
            add_pre_startup_function(f);
        }

        for (StartupFunctionType &f : global_startup_functions) {
            add_startup_function(f);
        }

        for (ShutdownFunctionType &f : global_pre_shutdown_functions) {
            add_pre_shutdown_function(f);
        }

        for (ShutdownFunctionType &f : global_shutdown_functions) {
            add_shutdown_function(f);
        }
    } catch (std::exception const &e) {
        // TODO: report_exception_and_terminate(e);
    } catch (...) {
        // TODO: report_exception_and_terminate(std::current_exception());
    }
}

void Runtime::add_pre_shutdown_function(ShutdownFunctionType f) {
    std::lock_guard l(_mutex);
    _pre_shutdown_functions.push_back(f);
}

void Runtime::add_shutdown_function(ShutdownFunctionType f) {
    std::lock_guard l(_mutex);
    _shutdown_functions.push_back(f);
}

void Runtime::add_pre_startup_function(StartupFunctionType f) {
    std::lock_guard l(_mutex);
    _pre_startup_functions.push_back(f);
}

void Runtime::add_startup_function(StartupFunctionType f) {
    std::lock_guard l(_mutex);
    _startup_functions.push_back(f);
}

int Runtime::run(std::function<EinsumsMainFunctionType> const &func) {
    // Once we start using a thread pool / threading manager we can
    // pass the function to the pool and have the manager handle it.
    return func();
}

} // namespace einsums::detail