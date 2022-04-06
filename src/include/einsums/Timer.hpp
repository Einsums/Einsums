#pragma once

#include <string>

namespace einsums::Timer {

void initialize();
void finalize();

void report();

void push(const std::string &name);
void pop();

struct Timer {
    Timer(const std::string &name) { push(name); }
    ~Timer() { pop(); }
};

} // namespace einsums::Timer