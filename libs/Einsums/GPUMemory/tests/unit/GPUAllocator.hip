
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::gpu;

TEST_CASE("Test allocator") {
    SECTION("Device only") {
        auto &config = GlobalConfigMap::get_singleton();

        auto alloc = GPUAllocator<double>();

        auto ptr = alloc.allocate(10);

        hipPointerAttribute_t attribs;

        hip_catch(hipPointerGetAttributes(&attribs, ptr));

        REQUIRE((attribs.type == hipMemoryTypeDevice));

        alloc.deallocate(ptr, 10);
    }

    SECTION("Mapped") {
        auto alloc = MappedAllocator<double>();

        auto ptr = alloc.allocate(10);

        hipPointerAttribute_t attribs;

        hip_catch(hipPointerGetAttributes(&attribs, ptr));

        REQUIRE((attribs.type == hipMemoryTypeHost));

        alloc.deallocate(ptr, 10);
    }
}

TEMPLATE_TEST_CASE("Test memcpy", "[gpu][memory]", double, std::complex<double>) {
    auto gpu_alloc = GPUAllocator<TestType>();

    auto random_dist = std::uniform_real_distribution<double>(-1, 1);

    int const len = 10;

    auto gpu_ptr1 = gpu_alloc.allocate(len);
    auto gpu_ptr2 = gpu_alloc.allocate(len);

    std::vector<TestType, BufferAllocator<TestType>> transfer(len), test(len);

    for (int i = 0; i < transfer.size(); i++) {
        if constexpr (IsComplexV<TestType>) {
            transfer[i] = {random_dist(einsums::random_engine), random_dist(einsums::random_engine)};
        } else {
            transfer[i] = random_dist(einsums::random_engine);
        }
    }

    std::memcpy(test.data(), transfer.data(), transfer.size() * sizeof(TestType));

    for (int i = 0; i < transfer.size(); i++) {
        REQUIRE(transfer[i] == test[i]);
    }

    std::memcpy(gpu_ptr1, transfer.data(), transfer.size() * sizeof(TestType));

    std::fill(transfer, 0.0);

    std::memcpy(transfer.data(), gpu_ptr1, transfer.size() * sizeof(TestType));

    for (int i = 0; i < transfer.size(); i++) {
        REQUIRE(transfer[i] == test[i]);
    }

    std::fill(transfer, 0.0);

    std::memcpy(gpu_ptr2, gpu_ptr1, transfer.size() * sizeof(TestType));

    std::memcpy(transfer.data(), gpu_ptr2, transfer.size() * sizeof(TestType));

    for (int i = 0; i < transfer.size(); i++) {
        REQUIRE(transfer[i] == test[i]);
    }

    gpu_alloc.deallocate(gpu_ptr1, len);
    gpu_alloc.deallocate(gpu_ptr2, len);
}