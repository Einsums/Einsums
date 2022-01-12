#pragma once

#include <string>

namespace EinsumsInCpp::Timer {

void initialize();
void finalize();

void report();

void push(const std::string &name);
void pop();

} // namespace EinsumsInCpp::Timer