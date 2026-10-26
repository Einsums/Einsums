//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <csignal>

#if defined(EINSUMS_WINDOWS)
#    include <Windows.h>
#endif

namespace einsums {
namespace detail {

EINSUMS_SINGLETON_IMPL(RuntimeVars)

RuntimeVars::RuntimeVars() {
    global_pre_startup_functions_  = new std::list<StartupFunctionType>();
    global_startup_functions_      = new std::list<StartupFunctionType>();
    global_pre_shutdown_functions_ = new std::list<ShutdownFunctionType>();
    global_shutdown_functions_     = new std::list<ShutdownFunctionType>();
}

RuntimeVars::~RuntimeVars() {
    std::fprintf(stderr, "Finalizing the RuntimeVars.\n");

    std::fprintf(stderr, "Freeing pre-startup functions.\n");
    if (global_pre_startup_functions_) {
        delete global_pre_startup_functions_;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }

    std::fprintf(stderr, "Freeing startup functions.\n");
    if (global_startup_functions_) {
        delete global_startup_functions_;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }

    std::fprintf(stderr, "Freeing pre-shutdown functions.\n");
    if (global_pre_shutdown_functions_) {
        delete global_pre_shutdown_functions_;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }

    std::fprintf(stderr, "Freeing shutdown functions.\n");
    if (global_shutdown_functions_) {
        delete global_shutdown_functions_;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }
}

void RuntimeVars::add_pre_startup_function(StartupFunctionType const &func) {
    global_pre_startup_functions_->push_back(func);
}

void RuntimeVars::add_startup_function(StartupFunctionType const &func) {
    global_startup_functions_->push_back(func);
}

void RuntimeVars::add_pre_shutdown_function(ShutdownFunctionType const &func) {
    global_pre_shutdown_functions_->push_back(func);
}

void RuntimeVars::add_shutdown_function(ShutdownFunctionType const &func) {
    global_shutdown_functions_->push_back(func);
}

std::list<StartupFunctionType> const &RuntimeVars::global_pre_startup_functions() const {
    return *global_pre_startup_functions_;
}

std::list<StartupFunctionType> const &RuntimeVars::global_startup_functions() const {
    return *global_startup_functions_;
}

std::list<ShutdownFunctionType> const &RuntimeVars::global_pre_shutdown_functions() const {
    return *global_pre_shutdown_functions_;
}

std::list<ShutdownFunctionType> const &RuntimeVars::global_shutdown_functions() const {
    return *global_shutdown_functions_;
}

static std::atomic_bool exit_called = false;

void on_exit() noexcept {
    exit_called = true;
}

void on_abort(int) noexcept {
    exit_called = true;
    std::exit(-1);
}

Runtime::Runtime(RuntimeConfiguration &&rtcfg, bool initialize) : _rtcfg(std::move(rtcfg)) {
    LabeledSectionInternal("Runtime constructor");
    init_global_data();

    if (initialize) {
        init();
    }
}

RuntimeState Runtime::state() const {
    return _state;
}

void Runtime::state(RuntimeState state) {
    EINSUMS_LOG_INFO("Runtime state changed from {} to {}", _state, state);
    _state = state;
}

RuntimeConfiguration &Runtime::config() {
    return _rtcfg;
}

RuntimeConfiguration const &Runtime::config() const {
    return _rtcfg;
}

void Runtime::init() {
    EINSUMS_LOG_INFO("Runtime::init: initializing...");
    try {
        /// @todo This would be a good place to create and initialize a thread pool

        auto                 &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard const vars_guard(runtime_vars); // Lock the variables.

        // Copy over all startup functions registered so far.
        for (StartupFunctionType const &f : runtime_vars.global_pre_startup_functions()) {
            add_pre_startup_function(f);
        }

        for (StartupFunctionType const &f : runtime_vars.global_startup_functions()) {
            add_startup_function(f);
        }

        for (ShutdownFunctionType const &f : runtime_vars.global_pre_shutdown_functions()) {
            add_pre_shutdown_function(f);
        }

        for (ShutdownFunctionType const &f : runtime_vars.global_shutdown_functions()) {
            add_shutdown_function(f);
        }
    } catch (std::exception const &e) {
        /// @todo report_exception_and_terminate(e);
    } catch (...) {
        /// @todo report_exception_and_terminate(std::current_exception());
    }
}

void Runtime::init_global_data() {
    LabeledSectionInternal("Runtime::init_global_data");
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(!runtime_);

    runtime_ = this;
}

void Runtime::deinit_global_data() {
    LabeledSectionInternal("Runtime::deinit_global_data");
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(runtime_);
    runtime_ = nullptr;
}

void Runtime::add_pre_shutdown_function(ShutdownFunctionType const &f) {
    LabeledSectionInternal("Runtime::add_pre_shutdown_function");
    std::lock_guard const l(this->lock_);
    _pre_shutdown_functions.push_back(f);
}

void Runtime::add_shutdown_function(ShutdownFunctionType const &f) {
    LabeledSectionInternal("Runtime::add_shutdown_function");
    std::lock_guard const l(this->lock_);
    _shutdown_functions.push_back(f);
}

void Runtime::add_pre_startup_function(StartupFunctionType const &f) {
    LabeledSectionInternal("Runtime::add_pre_startup_function");
    std::lock_guard const l(this->lock_);
    _pre_startup_functions.push_back(f);
}

void Runtime::add_startup_function(StartupFunctionType const &f) {
    LabeledSectionInternal("Runtime::add_startup_function");
    std::lock_guard const l(this->lock_);
    _startup_functions.push_back(f);
}

void Runtime::call_startup_functions(bool pre_startup) {
    if (pre_startup) {
        EINSUMS_LOG_TRACE("Calling pre-startup routines");
        LabeledSectionInternal("Calling pre-startup routines");
        state(RuntimeState::PreStartup);
        for (StartupFunctionType const &f : _pre_startup_functions) {
            f();
        }
    } else {
        EINSUMS_LOG_TRACE("Calling startup routines");
        LabeledSectionInternal("Calling startup routines");
        state(RuntimeState::Startup);
        for (StartupFunctionType const &f : _startup_functions) {
            f();
        }
    }
}

void Runtime::call_shutdown_functions(bool pre_shutdown) {
    if (pre_shutdown) {
        EINSUMS_LOG_TRACE("Calling pre-shutdown routines");
        LabeledSectionInternal("Calling pre-shutdown routines");
        state(RuntimeState::PreShutdown);
        for (ShutdownFunctionType const &f : _pre_shutdown_functions) {
            f();
        }
    } else {
        EINSUMS_LOG_TRACE("Calling shutdown routines");
        LabeledSectionInternal("Calling shutdown routines");
        state(RuntimeState::Shutdown);
        for (ShutdownFunctionType const &f : _shutdown_functions) {
            f();
        }
    }
}

int Runtime::run(std::function<EinsumsMainFunctionType> const &func) {
    call_startup_functions(true);
    call_startup_functions(false);

    // Set the state to running.
    state(RuntimeState::Running);
    // Once we start using a thread pool / threading manager we can
    // pass the function to the pool and have the manager handle it.
    EINSUMS_LOG_INFO("running user provided function");
    int result = func();

    return result;
}

int Runtime::run() {
    call_startup_functions(true);
    call_startup_functions(false);

    // Set the state to running.
    state(RuntimeState::Running);

    return 0;
}

} // namespace detail

bool is_running() {
    detail::Runtime *rt = runtime_ptr();
    if (nullptr != rt)
        return rt->state() == RuntimeState::Running;
    return false;
}

detail::Runtime &runtime() {
    EINSUMS_ASSERT(runtime_ptr() != nullptr);
    return *runtime_ptr();
}

detail::Runtime *&runtime_ptr() {
    static detail::Runtime *runtime_ = nullptr;
    return runtime_;
}

RuntimeConfiguration &runtime_config() {
    return runtime().config();
}

void register_pre_startup_function(StartupFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::PreStartup) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a pre-startup function");
            return;
        }
        runtime->add_pre_startup_function(f);
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.add_pre_startup_function(f);
    }
}

void register_startup_function(StartupFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::Startup) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a startup function");
            return;
        }
        runtime->add_startup_function(f);
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.add_startup_function(f);
    }
}

void register_pre_shutdown_function(ShutdownFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::PreShutdown) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a pre-shutdown function");
            return;
        }
        runtime->add_pre_shutdown_function(f);
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.add_pre_shutdown_function(f);
    }
}

void register_shutdown_function(ShutdownFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::Shutdown) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a shutdown function");
            return;
        }
        runtime->add_pre_shutdown_function(std::move(f));
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.add_shutdown_function(f);
    }
}

} // namespace einsums