#include "einsums/_Common.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/GPULinearAlgebra.hpp"

#include <catch2/catch_all.hpp>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

#include "catch2/catch_test_macros.hpp"
#include "einsums.hpp"
#include "src/backends/linear_algebra/hipblas/hipblas.hpp"

// Waste time for timing.
__global__ void time_waster(unsigned int max) {
    volatile float value;
    for (unsigned int i = 0; i < max; i++) {
        value = __sinf(i);
    }
}

#define KERNELS 20
#define CYCLES  0xffffU

TEST_CASE("GPU Timer", "[gpu][timer]") {

        einsums::timer::GPUTimer *timer = new einsums::timer::GPUTimer("GPU Timer");
        for (int i = 0; i < KERNELS; i++) {
            time_waster<<<dim3(256), dim3(2)>>>(CYCLES);
        }
        delete timer;
}