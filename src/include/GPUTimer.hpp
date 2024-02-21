//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Export.hpp"

#include <string>

namespace einsums::timer {

void EINSUMS_EXPORT gpu_initialize();
void EINSUMS_EXPORT gpu_finalize();

void EINSUMS_EXPORT gpu_report();

void EINSUMS_EXPORT gpu_push(std::string name);
void EINSUMS_EXPORT gpu_pop();

struct GPUTimer : public Timer {
    GPUTimer(const std::string &name) { push(name); }
    ~GPUTimer() { pop(); }
};

} // namespace einsums::timer