//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Debugging/AttachDebugger.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <csignal>
#include <spdlog/spdlog.h>

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

bool is_running() {
    Runtime *rt = runtime_ptr();
    if (nullptr != rt)
        return rt->state() == RuntimeState::Running;
    return false;
}

Runtime &runtime() {
    EINSUMS_ASSERT(runtime_ptr() != nullptr);
    return *runtime_ptr();
}

Runtime *&runtime_ptr() {
    static Runtime *runtime_ = nullptr;
    return runtime_;
}

Runtime::Runtime(RuntimeConfiguration &rtcfg, bool initialize) : _rtcfg(rtcfg) {
    init_global_data();

    if (initialize) {
        init();
    }
}

RuntimeState Runtime::state() const {
    return _state;
}

void Runtime::state(RuntimeState state) {
    spdlog::info("state change: from {} to {}", _state, state);
    _state = state;
}

RuntimeConfiguration &Runtime::config() {
    return _rtcfg;
}

RuntimeConfiguration const &Runtime::config() const {
    return _rtcfg;
}

void Runtime::init() {
    spdlog::info("Runtime::init: initializing...");
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

void Runtime::init_global_data() {
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(!runtime_);

    runtime_ = this;
}

void Runtime::deinit_global_data() {
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(runtime_);
    runtime_ = nullptr;
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

void Runtime::call_startup_functions(bool pre_startup) {
    if (pre_startup) {
        state(RuntimeState::PreStartup);
        for (StartupFunctionType &f : _pre_startup_functions) {
            f();
        }
    } else {
        state(RuntimeState::Startup);
        for (StartupFunctionType &f : _startup_functions) {
            f();
        }
    }
}

void Runtime::call_shutdown_functions(bool pre_shutdown) {
    if (pre_shutdown) {
        state(RuntimeState::PreShutdown);
        for (ShutdownFunctionType &f : _pre_shutdown_functions) {
            f();
        }
    } else {
        state(RuntimeState::Shutdown);
        for (ShutdownFunctionType &f : _shutdown_functions) {
            f();
        }
    }
}

int Runtime::run(std::function<EinsumsMainFunctionType> const &func) {
    call_startup_functions(true);
    spdlog::debug("run: ran pre-startup functions");

    call_startup_functions(false);
    spdlog::info("run: ran startup functions");

    // Set the state to running.
    state(RuntimeState::Running);
    // Once we start using a thread pool / threading manager we can
    // pass the function to the pool and have the manager handle it.
    spdlog::info("run: running user provided function");
    int result = func();

    return result;
}

int Runtime::run() {
    call_startup_functions(true);
    spdlog::debug("run: ran pre-startup functions");

    call_startup_functions(false);
    spdlog::info("run: ran startup functions");

    // Set the state to running.
    state(RuntimeState::Running);

    return 0;
}

} // namespace einsums::detail