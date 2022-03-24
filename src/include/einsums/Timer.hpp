#pragma once

#include <string>

namespace einsums::Timer {

void initialize();
void finalize();

void report();

void push(const std::string &name);
void pop();

} // namespace einsums::Timer