//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <chrono>
#include <string>

namespace einsums::profile {

using clock      = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<clock>;
using duration   = std::chrono::high_resolution_clock::duration;

namespace detail {
struct TimerDetail;
}

void EINSUMS_EXPORT initialize();
void EINSUMS_EXPORT finalize();

void EINSUMS_EXPORT report(std::string const &fname, bool append);

void EINSUMS_EXPORT push(std::string name);
void EINSUMS_EXPORT pop();
void EINSUMS_EXPORT pop(duration elapsed);

/**
 * @struct Timer
 *
 * Holds timing information for profiling Einsums.
 */
struct Timer {
  public:
    /**
     * Create a new timer with the given label.
     */
    explicit Timer(std::string const &name) {
        start = clock::now();
        push(name);
    }

    /**
     * Destroy the timer and update the timer stack.
     */
    ~Timer() {
        auto difference = clock::now() - start;
        pop(difference);
    }

  private:
    /**
     * @var start
     *
     * The time that the timer was started at.
     */
    time_point start;
};

} // namespace einsums::profile