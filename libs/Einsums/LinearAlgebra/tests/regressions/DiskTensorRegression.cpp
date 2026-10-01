//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Disk gemm regression", "[linear-algebra]", double, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    {
        auto &singleton = GlobalConfigMap::get_singleton();
        singleton.lock();
        singleton.set_string("buffer-size", "4GB");
        singleton.unlock();
    }

    constexpr size_t n = 700; // M > 500, not a multiple of 500 -- reproduces the bug

    Tensor<TestType, 2> A_core = create_random_tensor<TestType>("A", n, n);
    Tensor<TestType, 2> B_core = create_random_tensor<TestType>("B", n, n);

    Tensor<TestType, 2> C_baseline{"C baseline (in-core)", n, n};
    linear_algebra::gemm<false, false>(1.0, A_core, B_core, 0.0, &C_baseline);

    DiskTensor<TestType, 2> A_disk{fmt::format("/test/gemm2/{}/A", type_name<TestType>()), n, n};
    DiskTensor<TestType, 2> B_disk{fmt::format("/test/gemm2/{}/B", type_name<TestType>()), n, n};
    DiskTensor<TestType, 2> C_disk{fmt::format("/test/gemm2/{}/C", type_name<TestType>()), n, n};
    A_disk(All, All) = A_core;
    B_disk(All, All) = B_core;

    // NOTE: linear_algebra::gemm<false,false>(...) (the bool-template convenience overload) does
    // NOT compile for DiskTensor operands -- DiskAlgebra.hpp's disk-specific gemm() only has a
    // (char transA, char transB, ...) runtime-transpose overload, not a matching bool-template
    // one, so the top-level wrapper's detail::gemm<TransA,TransB>(...) call finds no viable
    // candidate. Using the char-based call directly instead (a separate, real API-completeness
    // gap worth noting).
    linear_algebra::gemm('n', 'n', 1.0, A_disk, B_disk, 0.0, &C_disk);

    auto  C_disk_view   = C_disk(All, All);  // keep the DiskView alive for the comparison loop
    auto &C_disk_result = C_disk_view.get(); // BufferTensor<double,2>, same element access API

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            CHECK_THAT(C_disk_result(i, j), einsums::CheckWithinRel(C_baseline(i, j), 1e-6));
        }
    }
}