#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>

#include "Einsums/Concepts/Complex.hpp"

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Test lassq", "[linear-algebra][gpu]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    constexpr int max_size = 2048;

    auto gpu_alloc = gpu::GPUAllocator<TestType>();

    auto gpu_buffer = gpu_alloc.allocate(max_size);

    {
        BufferVector<TestType> buffer(max_size);

        for (int i = 0; i < max_size; i++) {
            buffer[i] = TestType{1.0};
        }

        std::memcpy(gpu_buffer, buffer.data(), max_size * sizeof(TestType));
    }

    for (int i = 1; i < max_size; i++) {
        // int i = 64;
        RemoveComplexT<TestType> scale, sumsq;
        scale = 0.0;
        sumsq = 0.0;

        if constexpr (std::is_same_v<float, TestType>) {
            blas::hip::slassq(i, gpu_buffer, 1, &scale, &sumsq);
        } else if constexpr (std::is_same_v<double, TestType>) {
            blas::hip::dlassq(i, gpu_buffer, 1, &scale, &sumsq);
        } else if constexpr (std::is_same_v<std::complex<float>, TestType>) {
            blas::hip::classq(i, gpu_buffer, 1, &scale, &sumsq);
        } else if constexpr (std::is_same_v<std::complex<double>, TestType>) {
            blas::hip::zlassq(i, gpu_buffer, 1, &scale, &sumsq);
        }

        REQUIRE_THAT(scale * scale * sumsq, Catch::Matchers::WithinRel((RemoveComplexT<TestType>)i));
    }
}