#pragma once

#include "einsums/_Export.hpp"

#include <string>

namespace einsums::timer {

void EINSUMS_EXPORT initialize();
void EINSUMS_EXPORT finalize();

void EINSUMS_EXPORT report();

void EINSUMS_EXPORT push(const std::string &name);
void EINSUMS_EXPORT pop();

struct Timer {
    Timer(const std::string &name) { push(name); }
    ~Timer() { pop(); }
};

} // namespace einsums::timer