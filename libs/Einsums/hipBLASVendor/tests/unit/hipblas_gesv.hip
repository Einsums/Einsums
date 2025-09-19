#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>

#include "Einsums/Concepts/Complex.hpp"

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Test solver", "[linear-algebra][gpu]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    BufferVector<TestType> A{TestType{1.0}, TestType{4.0}, TestType{7.0}, TestType{2.0}, TestType{5.0},
                             TestType{8.0}, TestType{3.0}, TestType{6.0}, TestType{1.0}},
        B{TestType{11.0}, TestType{12.0}, TestType{13.0}},
        Expected{TestType{-31.0} / TestType{3.0}, TestType{32.0} / TestType{3.0}, TestType{0.0}};

    auto gpu_alloc = gpu::GPUAllocator<TestType>();
    auto int_alloc = gpu::GPUAllocator<int>();

    auto A_gpu = gpu_alloc.allocate(3 * 3);
    auto B_gpu = gpu_alloc.allocate(3);
    auto X_gpu = gpu_alloc.allocate(3);
    auto ipiv  = int_alloc.allocate(3);

    std::memcpy(A_gpu, A.data(), 3 * 3 * sizeof(TestType));
    std::memcpy(B_gpu, B.data(), 3 * sizeof(TestType));

    if constexpr (std::is_same_v<TestType, float>) {
        REQUIRE(blas::hip::sgesv(3, 1, A_gpu, 3, ipiv, B_gpu, 3, X_gpu, 3) == 0);
    } else if constexpr (std::is_same_v<TestType, double>) {
        REQUIRE(blas::hip::dgesv(3, 1, A_gpu, 3, ipiv, B_gpu, 3, X_gpu, 3) == 0);
    } else if constexpr (std::is_same_v<TestType, std::complex<float>>) {
        REQUIRE(blas::hip::cgesv(3, 1, A_gpu, 3, ipiv, B_gpu, 3, X_gpu, 3) == 0);
    } else {
        REQUIRE(blas::hip::zgesv(3, 1, A_gpu, 3, ipiv, B_gpu, 3, X_gpu, 3) == 0);
    }

    std::memcpy(B.data(), X_gpu, 3 * sizeof(TestType));

    gpu_alloc.deallocate(A_gpu, 3 * 3);
    gpu_alloc.deallocate(B_gpu, 3);
    gpu_alloc.deallocate(X_gpu, 3);
    int_alloc.deallocate(ipiv, 3);

    if constexpr (IsComplexV<TestType>) {
        for (int i = 0; i < 3; i++) {
            REQUIRE_THAT(std::real(B[i]), Catch::Matchers::WithinAbs(std::real(Expected[i]), 1e-3));
            REQUIRE_THAT(std::imag(B[i]), Catch::Matchers::WithinAbs(std::imag(Expected[i]), 1e-3));
        }
    } else {
        for (int i = 0; i < 3; i++) {
            REQUIRE_THAT(B[i], Catch::Matchers::WithinAbs(Expected[i], 1e-3));
        }
    }
}