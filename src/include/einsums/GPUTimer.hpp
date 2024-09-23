//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Export.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/Timer.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <string>

namespace einsums::timer {

/**
 * @struct GPUTimer
 *
 * @brief Timer for GPU kernels.
 *
 * Timing for GPU kernels is a bit different than normal. An event needs to be pushed to a stream before a
 * kernel runs. Then, an event is pushed to the stream after a kernel is pushed. When the events reach the
 * head of the stream, they record their time. This allows for kernel timing.
 */
struct EINSUMS_EXPORT GPUTimer {
  private:
    hipEvent_t start_event, end_event;

  public:
    GPUTimer(const std::string &name);
    ~GPUTimer();
};

} // namespace einsums::timer