//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/Debug.hpp>
#include <Einsums/Config/StdAlternatives.hpp>
#ifdef EINSUMS_HAVE_BACKTRACES
#    include <Einsums/Debugging/Backtrace.hpp>

#    include <cpptrace/cpptrace.hpp>
#endif
#include <Einsums/Runtime.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/internal/catch_context.hpp>
#include <functional>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

#ifdef EINSUMS_WINDOWS
#    include <fmt/format.h>
#    include <fmt/xchar.h>

#    include <Windows.h>
#    include <cerrno>
#    include <corecrt.h>
#    include <cstdio>
#    include <cwchar>

extern "C" void __cdecl einsums_invalid_parameter(wchar_t const *const expression, wchar_t const *const function_name,
                                                  wchar_t const *const file_name, unsigned int const line_number,
                                                  uintptr_t const reserved) {
    errno = EINVAL;

    std::fputs(einsums::util::backtrace().c_str(), stderr);

    if (expression != nullptr && function_name != nullptr && file_name != nullptr) {
        std::fputws(fmt::format(L"Einsums test: Error in {} at {}:{}: {}", function_name, file_name, line_number, expression).c_str(),
                    stderr);
    } else {
        std::fputs("Error at unknown location!", stderr);
    }
    std::fflush(stderr);
}
#endif

int einsums_main(int argc, char *const *const argv) {
    int result;
#pragma omp parallel
    {
#pragma omp single
        {
            einsums::GlobalConfigMap::get_singleton().set_string("profiler-filename",
                                                              einsums::detail::corrected_format("profile.{}.txt", einsums::getpid()));

            Catch::Session session;
            session.applyCommandLine(argc, argv);

            Catch::StringMaker<float>::precision  = std::numeric_limits<float>::digits10;
            Catch::StringMaker<double>::precision = std::numeric_limits<double>::digits10;
            auto seed                             = session.config().rngSeed();

#pragma omp parallel
            {
                einsums::random_engine().seed(seed);
            }

            result = session.run();
            einsums::finalize();
        }
    }
    return result;
}

int main(int argc, char **argv) {
#ifdef EINSUMS_WINDOWS
    auto prev_handler = _set_invalid_parameter_handler(einsums_invalid_parameter);

#    ifdef EINSUMS_DEBUG
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#    endif
#endif

    return einsums::start(einsums_main, argc, argv);
}
