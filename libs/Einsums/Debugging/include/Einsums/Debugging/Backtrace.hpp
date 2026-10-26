//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>
#include <string>

#if defined(EINSUMS_HAVE_BACKTRACES)
#    include <cpptrace/cpptrace.hpp>

namespace einsums::util {

/**
 * @brief Generate a backtrace.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT std::string backtrace(std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH);

/**
 * @brief Print a backtrace directly to a stream without having to generate an intermediate string on our end.
 *
 * @param stream The stream to print to. Just needs to be an @c ostream
 * @param frames_no The number of frames. The default value is configured at compile time.
 *
 * @versionadded{1.1.6}
 */
template <typename Ostream>
void print_backtrace(Ostream &stream, std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH) {
    auto trace = cpptrace::generate_trace(1, frames_no);

    stream << trace;
}

} // namespace einsums::util

#else

namespace einsums::util {

/**
 * @brief Generate a backtrace.
 *
 * @versionadded{1.0.0}
 */
inline std::string backtrace(std::size_t frames_no = 0) {
    return "";
}

/**
 * @brief Print a backtrace directly to a stream without having to generate an intermediate string on our end.
 *
 * @param stream The stream to print to. Just needs to be an @c ostream
 * @param frames_no The number of frames. The default value is configured at compile time.
 *
 * @versionadded{1.1.6}
 */
template <typename Ostream>
void print_backtrace(Ostream &stream, std::size_t frames_no = 0) {
    ; // noop
}

} // namespace einsums::util

#endif
