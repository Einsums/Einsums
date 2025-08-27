//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <chrono>
#include <string>

namespace einsums::profile {

/**
 * @typedef clock
 *
 * The type used for the clock.
 *
 * @versionadded{1.0.0}
 */
using clock = std::chrono::high_resolution_clock;

/**
 * @typedef time_point
 *
 * The time point type for the clock.
 *
 * @versionadded{1.0.0}
 */
using time_point = std::chrono::time_point<clock>;

/**
 * @typedef duration
 *
 * The duration type for the clock.
 *
 * @versionadded{1.0.0}
 */
using duration = std::chrono::high_resolution_clock::duration;

namespace detail {
struct TimerDetail;
}

/**
 * Initialize the timer.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT initialize();

/**
 * Finalize the timer.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT finalize();

/**
 * Report the profilng info.
 *
 * @param[in] fname The file to print to. If it does not exist, it will be created.
 * @param[in] append If true, the current timer report will be appended to the end of the file. If false, then the file will be overwritten.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT report(std::string const &fname, bool append);

/**
 * Push a timer with the given name. If the timer already exists, the data will be added to that timer.
 *
 * @param[in] name The name for the new timer.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT push(std::string name);

/**
 * Pop the current timer.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT pop();

/**
 * Pop the current timer and tell it how long it took.
 *
 * @param[in] elapsed The time taken for the profiled operation.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT pop(duration elapsed);

/**
 * @struct Timer
 *
 * Holds timing information for profiling Einsums.
 *
 * @versionadded{1.0.0}
 */
struct Timer {
  public:
    /**
     * Create a new timer with the given label.
     *
     * @param[in] name The label for the timer.
     *
     * @versionadded{1.0.0}
     */
    inline explicit Timer(std::string const &name) {
        start = clock::now();
        push(name);
    }

    /**
     * Destroy the timer and update the timer stack.
     *
     * @versionadded{1.0.0}
     */
    inline ~Timer() {
        auto difference = clock::now() - start;
        pop(difference);
    }

  private:
    /**
     * @var start
     *
     * The time that the timer was started at.
     *
     * @versionadded{1.0.0}
     */
    time_point start;
};

} // namespace einsums::profile