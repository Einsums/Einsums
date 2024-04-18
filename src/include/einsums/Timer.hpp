//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Export.hpp"

#include <chrono>
#include <string>

namespace einsums::timer {

using clock      = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<clock>;
using duration   = std::chrono::high_resolution_clock::duration;

namespace detail {
struct TimerDetail;
}

void EINSUMS_EXPORT initialize();
void EINSUMS_EXPORT finalize();

void EINSUMS_EXPORT report();
void EINSUMS_EXPORT report(const std::string &fname);
// void EINSUMS_EXPORT report(const char *fname); // const std::string& should be able to handle this case.
void EINSUMS_EXPORT report(std::FILE *fp);
void EINSUMS_EXPORT report(std::ostream &os);

void EINSUMS_EXPORT push(std::string name);
void EINSUMS_EXPORT pop();
void EINSUMS_EXPORT pop(duration elapsed);

struct Timer {
  private:
    time_point start;

  public:
    explicit Timer(const std::string &name) {
        start = clock::now();
        push(name);
    }

    ~Timer() {
        auto difference = clock::now() - start;
        pop(difference);
    }
};

} // namespace einsums::timer