//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Runtime/StartupFunction.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

#include <list>
#include <mutex>

namespace einsums {

enum class RuntimeState : std::int8_t {
    invalid          = -1,
    initialized      = 0,
    pre_startup      = 1,
    startup          = 2,
    pre_main         = 3,
    starting         = 4,
    running          = 5,
    pre_shutdown     = 6,
    shutdown         = 7,
    stopping         = 8,
    terminating      = 9,
    stopped          = 10,
    last_valid_state = stopped,
};

namespace detail {

extern std::list<StartupFunctionType>  global_pre_startup_functions;
extern std::list<StartupFunctionType>  global_startup_functions;
extern std::list<ShutdownFunctionType> global_pre_shutdown_functions;
extern std::list<ShutdownFunctionType> global_shutdown_functions;

struct EINSUMS_EXPORT Runtime {
    virtual ~Runtime() = default;

    /// The \a EinsumsMainFunctionType is the default function type used as
    /// the main Einsums function.
    using EinsumsMainFunctionType = int();

    /// Construct a new Einsums runtime instance
    Runtime(RuntimeConfiguration &rtcfg, bool initialize);

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

  protected:
    /// Common initialization for different constructors
    void init();

  private:
    std::list<StartupFunctionType>  _pre_startup_functions;
    std::list<StartupFunctionType>  _startup_functions;
    std::list<ShutdownFunctionType> _pre_shutdown_functions;
    std::list<ShutdownFunctionType> _shutdown_functions;

  protected:
    mutable std::mutex   _mutex;
    RuntimeConfiguration _rtcfg;

    std::atomic<RuntimeState> _state;
};

EINSUMS_EXPORT void on_exit() noexcept;
EINSUMS_EXPORT void on_abort(int signal) noexcept;

} // namespace detail
} // namespace einsums