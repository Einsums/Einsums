
#include <Einsums/GPUMemory/GPUAllocator.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::gpu;

TEST_CASE("Test allocator") {
    SECTION("Device only") {
        auto alloc = GPUAllocator<double>();

        auto *ptr = alloc.allocate(10);

        hipPointerAttribute_t attribs;

        hip_catch(hipPointerGetAttributes(&attribs, ptr));

        REQUIRE((attribs.type == hipMemoryTypeDevice));

        alloc.deallocate(ptr, 10);
    }

    SECTION("Device only") {
        auto alloc = MapAllocator<double>();

        auto *ptr = alloc.allocate(10);

        hipPointerAttribute_t attribs;

        hip_catch(hipPointerGetAttributes(&attribs, ptr));

        REQUIRE((attribs.type == hipMemoryTypeHost));

        alloc.deallocate(ptr, 10);
    }
}