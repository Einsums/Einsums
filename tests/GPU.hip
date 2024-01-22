#include "einsums.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <type_traits>



TEST_CASE("GPU Tensor Creation", "[gpu]") {
    using namespace einsums;
    using namespace einsums::gpu;

    DeviceTensor<double, 2> A("A", einsums::gpu::detail::DEV_ONLY, 3, 3);
}