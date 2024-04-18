#include "einsums/GPUTimer.hpp"
#include <chrono>
#include <hip/hip_runtime_api.h>
#include <hip/hip_common.h>

namespace einsums::timer {

GPUTimer::GPUTimer(const std::string &name) { 
        using namespace einsums::gpu;
        hip_catch(hipEventCreate(&start_event));
        hip_catch(hipEventCreate(&end_event));

        hip_catch(hipEventRecord(start_event, get_stream()));

        push(name);
    }
GPUTimer::~GPUTimer() {
        using namespace einsums::gpu;
        hip_catch(hipEventRecord(end_event, get_stream()));

        hip_catch(hipEventSynchronize(end_event));

        float diff;

        hip_catch(hipEventElapsedTime(&diff, start_event, end_event));

        pop(std::chrono::microseconds((long) (1000 * diff)));

        hip_catch(hipEventDestroy(start_event));
        hip_catch(hipEventDestroy(end_event));
    }

}