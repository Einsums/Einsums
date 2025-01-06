//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Runtime/StartupFunction.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

#include <list>
#include <mutex>
#include <string_view>

namespace einsums {

/**
 * @struct invalid_runtime_state
 *
 * Indicates that the code is handling data that is uninitialized.
 */
struct EINSUMS_EXPORT invalid_runtime_state : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class RuntimeState : std::int8_t {
    Invalid        = -1,
    Initialized    = 0,
    PreStartup     = 1,
    Startup        = 2,
    PreMain        = 3,
    Starting       = 4,
    Running        = 5,
    PreShutdown    = 6,
    Shutdown       = 7,
    Stopping       = 8,
    Terminating    = 9,
    Stopped        = 10,
    LastValidState = Stopped,
};

namespace detail {

EINSUMS_EXPORT std::list<StartupFunctionType> &global_pre_startup_functions();
EINSUMS_EXPORT std::list<StartupFunctionType> &global_startup_functions();
EINSUMS_EXPORT std::list<ShutdownFunctionType> &global_pre_shutdown_functions();
EINSUMS_EXPORT std::list<ShutdownFunctionType> &global_shutdown_functions();

struct EINSUMS_EXPORT Runtime {
    virtual ~Runtime() = default;

    /// The \a EinsumsMainFunctionType is the default function type used as
    /// the main Einsums function.
    using EinsumsMainFunctionType = int();

    /// Construct a new Einsums runtime instance
    Runtime(RuntimeConfiguration &&rtcfg, bool initialize);

    RuntimeState state() const;
    void         state(RuntimeState s);

    RuntimeConfiguration       &config();
    RuntimeConfiguration const &config() const;

    /// Add a function to be executed before einsums_main
    /// but guaranteed to be executed before any startup function registered
    /// with \a add_startup_function.
    ///
    /// \param  f   The function 'f' will be called  before pika_main is executed. This is very useful
    ///             to setup the runtime environment of the application
    ///             (install performance counters, etc.)
    ///
    /// \note       The difference to a startup function is that all
    ///             pre-startup functions will be (system-wide) executed
    ///             before any startup function.
    virtual void add_pre_startup_function(StartupFunctionType f);

    /// Add a function to be executed before einsums_main
    ///
    /// \param  f   The function 'f' will be called before einsums_main is executed. This is very useful
    ///             to setup the runtime environment of the application
    ///             (install performance counters, etc.)
    virtual void add_startup_function(StartupFunctionType f);

    /// Add a function to be executed during
    /// einsums::finalize, but guaranteed before any of the shutdown functions
    /// is executed.
    ///
    /// \param  f   The function 'f' will be called while einsums::finalize is executed. This is very
    ///             useful to tear down the runtime environment of the
    ///             application (uninstall performance counters, etc.)
    ///
    /// \note       The difference to a shutdown function is that all
    ///             pre-shutdown functions will be (system-wide) executed
    ///             before any shutdown function.
    virtual void add_pre_shutdown_function(ShutdownFunctionType f);

    /// Add a function to be executed during einsums::finalize
    ///
    /// \param  f   The function 'f' will be called while einsums::finalize is executed. This is very
    ///             useful to tear down the runtime environment of the
    ///             application (uninstall performance counters, etc.)
    virtual void add_shutdown_function(ShutdownFunctionType f);

    virtual int run(std::function<EinsumsMainFunctionType> const &func);
    virtual int run();

  protected:
    /// Common initialization for different constructors
    void init();
    void init_global_data();
    void deinit_global_data();

  private:
    void call_startup_functions(bool pre_startup);
    void call_shutdown_functions(bool pre_shutdown);

    friend void einsums::finalize();

    std::list<StartupFunctionType>  _pre_startup_functions;
    std::list<StartupFunctionType>  _startup_functions;
    std::list<ShutdownFunctionType> _pre_shutdown_functions;
    std::list<ShutdownFunctionType> _shutdown_functions;

  protected:
    mutable std::mutex   _mutex;
    RuntimeConfiguration _rtcfg;

    std::atomic<RuntimeState> _state{RuntimeState::Invalid};
};

EINSUMS_EXPORT void on_exit() noexcept;
EINSUMS_EXPORT void on_abort(int signal) noexcept;
EINSUMS_EXPORT void set_signal_handlers();
} // namespace detail

/// The function \a get_runtime returns a reference to the (thread
/// specific) runtime instance.
EINSUMS_EXPORT detail::Runtime &runtime();
EINSUMS_EXPORT detail::Runtime *&runtime_ptr();

EINSUMS_EXPORT RuntimeConfiguration &runtime_config();

///////////////////////////////////////////////////////////////////////////
/// \brief Test whether the runtime system is currently running.
///
/// This function returns whether the runtime system is currently running
/// or not, e.g.  whether the current state of the runtime system is
/// \a einsums::RuntimeState::Running
///
/// \note   This function needs to be executed on a pika-thread. It will
///         return false otherwise.
EINSUMS_EXPORT bool is_running();

} // namespace einsums

template <>
struct fmt::formatter<einsums::RuntimeState> : formatter<string_view> {
    template <typename FormatContext>
    auto format(einsums::RuntimeState state, FormatContext &ctx) const {
        std::string_view name;
        switch (state) {
        case einsums::RuntimeState::Invalid:
            name = "Invalid";
            break;
        case einsums::RuntimeState::Initialized:
            name = "Initialized";
            break;
        case einsums::RuntimeState::PreStartup:
            name = "PreStartup";
            break;
        case einsums::RuntimeState::Startup:
            name = "Startup";
            break;
        case einsums::RuntimeState::PreMain:
            name = "PreMain";
            break;
        case einsums::RuntimeState::Starting:
            name = "Starting";
            break;
        case einsums::RuntimeState::Running:
            name = "Running";
            break;
        case einsums::RuntimeState::PreShutdown:
            name = "PreShutdown";
            break;
        case einsums::RuntimeState::Shutdown:
            name = "Shutown";
            break;
        case einsums::RuntimeState::Stopping:
            name = "Stopping";
            break;
        case einsums::RuntimeState::Terminating:
            name = "Terminating";
            break;
        case einsums::RuntimeState::Stopped:
            name = "Stopped";
            break;
        default:
            name = "Unknown";
        }
        return formatter<string_view>::format(name, ctx);
    }
};