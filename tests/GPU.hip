#include "einsums/_Common.hpp"
#include "einsums/DeviceTensor.hpp"
#include "einsums/GPULinearAlgebra.hpp"
#include "catch2/catch_test_macros.hpp"
#include "einsums.hpp"
#include "src/backends/linear_algebra/hipblas/hipblas.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>

#include <catch2/catch_all.hpp>
#include <type_traits>



TEST_CASE("GPU Tensor Creation", "[gpu]") {
    using namespace einsums;
    using namespace einsums::gpu;

    gpu::initialize();


    DeviceTensor<double, 2> A("A", einsums::gpu::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::gpu::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::gpu::detail::PINNED, 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    for(size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < 3; j++) {
            std::fprintf(stderr, "%f ", A(i, j).get());
        }
    }

    std::fprintf(stderr, "\n");

    A.zero();
    B.zero();
    C.zero();

    for(size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < 3; j++) {
            std::fprintf(stderr, "%f ", A(i, j).get());
        }
    }

    for(size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < 3; j++) {
            REQUIRE(A(i, j).get() == 0);
            REQUIRE(B(i, j).get() == 0);
            REQUIRE(C(i, j).get() == 0);
        }
    }

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    for(size_t i = 0; i < 3; i++) {
        REQUIRE(A(i, i).get() == 1.0);
        REQUIRE(B(i, i).get() == 1.0);
        for(size_t j = 0; j < i; j++) {
            REQUIRE(A(i, j).get() == 0);
            REQUIRE(B(i, j).get() == 0);
            REQUIRE(A(j, i).get() == 0);
            REQUIRE(B(j, i).get() == 0);
        }
    }

    einsums::backend::linear_algebra::hipblas::initialize();

    einsums::gpu::linear_algebra::gemm<false, false>(1.0, A, B, 1.0, &C);

    for(size_t i = 0; i < 3; i++) {
        REQUIRE(C(i, i).get() == 1.0);
        for(size_t j = 0; j < i; j++) {
            REQUIRE(C(i, j).get() == 0);
            REQUIRE(C(j, i).get() == 0);
        }
    }

    einsums::backend::linear_algebra::hipblas::finalize();
}