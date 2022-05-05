#include "einsums/Blas.hpp"
#include "einsums/LinearAlgebra.hpp"
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
    blas::initialize();

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

    Tensor A("A", 3, 3);
    Tensor B("B", 3, 3);
    Tensor C("C", 3, 3);

    A.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    B.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>{11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};

    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    println(C);
    einsums::linear_algebra::gemm<true, false>(1.0, A, B, 0.0, &C);
    println(C);
    einsums::linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C);
    println(C);
    einsums::linear_algebra::gemm<true, true>(1.0, A, B, 0.0, &C);
    println(C);

    timer::report();

    blas::finalize();
    timer::finalize();

    return 0;
}