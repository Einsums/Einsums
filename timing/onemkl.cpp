#include "einsums/Print.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Timer.hpp"

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

auto main() -> int {
    ////////////////////////////////////
    // Form the two-electron integrals//
    ////////////////////////////////////
    using namespace einsums;

    timer::initialize();

    auto platforms = sycl::platform::get_platforms();
    for (auto &platform : platforms) {
        println("Platform: {} {}", platform.get_info<sycl::info::platform::name>(), platform.is_host());

        auto devices = platform.get_devices();
        for (auto &device : devices) {
            println("  Device: {}", device.get_info<sycl::info::device::name>());
        }
    }

    sycl::queue Q;
    println("Selected default queue uses {}", Q.get_device().get_info<sycl::info::device::name>());
    println("Host selector {}", sycl::device(sycl::host_selector()).get_info<sycl::info::device::name>());

    timer::report();
    timer::finalize();

    return 0;
}