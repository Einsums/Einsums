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
#include <Einsums/Runtime/Detail/InitLogging.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <csignal>
#include <cstdlib>
#include <functional>
#include <h5cpp/all>
#include <spdlog/spdlog.h>
#include <tuple>
#include <unordered_map>

namespace einsums {

template <typename F, typename... BoundArgs>
struct bind_back_t {
    F                        func_;
    std::tuple<BoundArgs...> bound_args_;

    template <typename... CallArgs, std::size_t... Indices>
    auto invoke_impl(std::index_sequence<Indices...>, CallArgs &&...call_args) const {
        return func_(std::forward<CallArgs>(call_args)..., std::get<Indices>(bound_args_)...);
    }

    template <typename... CallArgs, std::size_t... Indices>
    auto invoke_impl(std::index_sequence<Indices...>, CallArgs &&...call_args) {
        return func_(std::forward<CallArgs>(call_args)..., std::get<Indices>(bound_args_)...);
    }

    template <typename Func, typename... Args>
    bind_back_t(Func &&func, Args &&...args) : func_(std::forward<Func>(func)), bound_args_(std::forward<Args>(args)...) {}

    template <typename... CallArgs>
    auto operator()(CallArgs &&...call_args) const {
        return invoke_impl(std::index_sequence_for<BoundArgs...>{}, std::forward<CallArgs>(call_args)...);
    }

    template <typename... CallArgs>
    auto operator()(CallArgs &&...call_args) {
        return invoke_impl(std::index_sequence_for<BoundArgs...>{}, std::forward<CallArgs>(call_args)...);
    }
};

template <typename F, typename... BoundArgs>
auto bind_back(F &&func, BoundArgs &&...bound_args) {
    return bind_back_t<std::decay_t<F>, std::decay_t<BoundArgs>...>(std::forward<F>(func), std::forward<BoundArgs>(bound_args)...);
}

namespace detail {

void add_startup_functions(Runtime &rt, RuntimeConfiguration const &cfg, StartupFunctionType startup, ShutdownFunctionType shutdown) {
    if (!!startup) {
        rt.add_startup_function(std::move(startup));
    }

    if (!!shutdown) {
        rt.add_shutdown_function(std::move(shutdown));
    }
}

int run(std::function<int()> const &f, Runtime &rt, InitParams const &params) {
    add_startup_functions(rt, rt.config(), std::move(params.startup), std::move(params.shutdown));

    // Run this runtime instance using the given function f
    if (f) {
        return rt.run(f);
    }

    // Run this runtime instance without an einsums_main
    return rt.run();
}

int run(std::function<int()> const &f, int argc, char const *const *argv, InitParams const &params, bool blocking) {
    EINSUMS_LOG_INFO("Running common initialization routines...");
    // TODO: Add a check to ensure the runtime hasn't already been initialized

    // Command line arguments for Einsums will be prefixed with --einsums:
    // For example, "--einsums:verbose=1" will be translated to verbose=1
    std::unordered_map<std::string, std::string> cmdline;
    RuntimeConfiguration                         config(argc, argv);

    // Before this line logging does not work.
    init_logging(config);

    if (config.einsums.install_signal_handlers) {
        EINSUMS_LOG_TRACE("Installing signal handlers...");
        set_signal_handlers();
    }

    // This is the only initialization routine that needs to be explicitly called here.
    // This is because the runtime environment depends on the profiler. If the profiler
    // depended on the runtime environment, then there would be a dependency issue.
    profile::initialize();

    // Disable HDF5 diagnostic reporting
    H5Eset_auto(0, nullptr, nullptr);

    // Build and configure this runtime instance.
    std::unique_ptr<Runtime> rt = std::make_unique<Runtime>(std::move(config), true);

    if (blocking) {
        return run(f, *rt, params);
    }

    run(f, *rt, params);

    // pointer to runtime is stored in TLS
    [[maybe_unused]] Runtime *p = rt.release();

    return 0;
}

int run_impl(std::function<int()> f, int argc, char const *const *argv, InitParams const &params, bool blocking) {
    if (argc == 0 || argv == nullptr) {
        argc = dummy_argc;
        argv = dummy_argv;
    }

    // register default handlers
    [[maybe_unused]] auto signal_handler = std::signal(SIGABRT, on_abort);
    [[maybe_unused]] auto exit_result    = std::atexit(on_exit);
#if defined(EINSUMS_HAVE_CXX11_STD_QUICK_EXIT)
    [[maybe_unused]] auto quick_exit_result = std::at_quick_exit(on_exit);
#endif
    return run(f, argc, argv, params, blocking);
}

} // namespace detail

int start(std::function<int()> f, int argc, char **argv, InitParams const &params) {
    return detail::run_impl(std::move(f), argc, argv, params, true);
}

int start(std::function<int(int, char **)> f, int argc, char **argv, InitParams const &params) {
    // So doing this the user function "f" will receive einsums specific command line parameters too
    // If they are not expecting that they may produce an error.
    //
    // We should bind through a helper function that will take the constructed RuntimeConfiguration's
    // post-processed command-line options (einsums options filtered out) and pass that to the
    // user function.
    std::function<int()> main_f = std::bind(f, argc, argv);
    return detail::run_impl(std::move(main_f), argc, argv, params, true);
}

int start(std::nullptr_t, int argc, char **argv, InitParams const &params) {
    std::function<int()> main_f;
    return detail::run_impl(std::move(main_f), argc, argv, params, true);
}

void initialize(std::function<int()> f, int argc, char const *const *argv, InitParams const &params) {
    std::function<int()> main_f = std::bind(f);
    if (detail::run_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(std::nullptr_t, int argc, char const *const *argv, InitParams const &params) {
    std::function<int()> main_f;
    if (detail::run_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

void initialize(int argc, char const *const *argv, InitParams const &params) {
    std::function<int()> main_f;
    if (detail::run_impl(std::move(main_f), argc, argv, params, false) != 0) {
        EINSUMS_UNREACHABLE;
    }
}

} // namespace einsums