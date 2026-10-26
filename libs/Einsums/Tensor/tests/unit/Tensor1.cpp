//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Tensor creation", "[tensor]", float, double) {
    using namespace einsums;

    Tensor<TestType, 2> A(true, "A", 1, 1);
    Tensor<TestType, 2> B(true, "B", 1, 1);

    REQUIRE((A.dim(0) == 1 && A.dim(1) == 1));
    REQUIRE((B.dim(0) == 1 && B.dim(1) == 1));

    A.resize(einsums::Dim{3, 3});
    B.resize(einsums::Dim{3, 3});

    auto C = create_tensor<TestType>(true, "C", 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    A.zero();
    B.zero();
    C.zero();

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    CHECK_THAT(B.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));

    // Set A and B to identity
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    CHECK_THAT(B.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(std::vector<TestType>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
}

// TEST_CASE("Tensor-2D HDF5") {
//     using namespace einsums;

//     einsums::Tensor A("A", 3, 3);

//     for (int i = 0, ij = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++, ij++) {
//             A(i, j) = ij;
//         }
//     }

//     {
//         h5::fd_t fd = h5::create("tensor.h5", H5F_ACC_TRUNC);
//         h5::ds_t ds = h5::create<double>(fd, "Matrix A", h5::current_dims{10, 20}, h5::max_dims{10, 20});
//         h5::write(ds, A, h5::count{2, 3}, h5::offset{2, 2}, h5::stride{1, 3});
//     }

//     {
//         auto B = h5::read<einsums::Tensor<double, 2>>("tensor.h5", "Matrix A");

//         REQUIRE((B.dim(0) == 10 && B.dim(1) == 20));
//         REQUIRE(B(2, 2) == 0.0);
//         REQUIRE(B(2, 5) == 1.0);
//         REQUIRE(B(2, 8) == 2.0);
//         REQUIRE(B(3, 2) == 3.0);
//         REQUIRE(B(3, 5) == 4.0);
//         REQUIRE(B(3, 8) == 5.0);
//     }
// }

// TEST_CASE("Tensor-1D HDF5") {
//     auto A = einsums::create_random_tensor("A", 3);

//     {
//         h5::fd_t fd = h5::create("tensor-1d.h5", H5F_ACC_TRUNC);
//         h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3}, h5::max_dims{3});
//         h5::write(ds, A);
//     }

//     {
//         auto B = h5::read<einsums::Tensor<double, 1>>("tensor-1d.h5", "A");

//         REQUIRE(A(0) == B(0));
//         REQUIRE(A(1) == B(1));
//         REQUIRE(A(2) == B(2));
//     }
// }

// TEST_CASE("Tensor-3D HDF5") {
//     auto A = einsums::create_random_tensor("A", 3, 2, 1);

//     {
//         h5::fd_t fd = h5::create("tensor-3d.h5", H5F_ACC_TRUNC);
//         h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3, 2, 1});
//         h5::write(ds, A);
//     }

//     {
//         auto B = h5::read<einsums::Tensor<double, 3>>("tensor-3d.h5", "A");

//         REQUIRE(B.dim(0) == 3);
//         REQUIRE(B.dim(1) == 2);
//         REQUIRE(B.dim(2) == 1);
//     }
// }

// TEST_CASE("TensorView-2D HDF5") {
//     SECTION("Subview Offset{0,0,0}") {
//         auto                           A = einsums::create_random_tensor("A", 3, 3, 3);
//         einsums::TensorView<double, 2> viewA(A, einsums::Dim{3, 9});

//         REQUIRE((A.dim(0) == 3 && A.dim(1) == 3 && A.dim(2) == 3));
//         REQUIRE((viewA.dim(0) == 3 && viewA.dim(1) == 9));

//         {
//             h5::fd_t fd = h5::create("tensorview-2d.h5", H5F_ACC_TRUNC);
//             h5::ds_t ds = h5::create<double>(fd, "A", h5::current_dims{3, 9});
//             h5::write(ds, viewA);
//         }

//         {
//             auto B = h5::read<einsums::Tensor<double, 2>>("tensorview-2d.h5", "A");
//             for (int i = 0; i < 3; i++)
//                 for (int j = 0; j < 9; j++)
//                     REQUIRE(viewA(i, j) == B(i, j));
//         }
//     }
// }



// TEST_CASE("Tensor 2D - HDF5 wrapper") {
//     auto A = einsums::create_random_tensor("A", 3, 3);

//     h5::fd_t fd = h5::create("tensor-wrapper.h5", H5F_ACC_TRUNC);

//     einsums::write(fd, A);

//     auto B = einsums::read<double, 2>(fd, "A");

//     for (int i = 0; i < 3; i++)
//         for (int j = 0; j < 3; j++)
//             REQUIRE(A(i, j) == B(i, j));
// }
