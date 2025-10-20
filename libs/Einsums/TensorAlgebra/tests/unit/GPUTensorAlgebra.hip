//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Testing.hpp>
#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateIdentity.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>

TEST_CASE("GPU Identity Tensor", "[tensor]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    DeviceTensor<double, 2> I("I", einsums::detail::DEV_ONLY, 3, 3);
    I = create_identity_tensor("I", 3, 3);

    REQUIRE((double)I(0, 0) == 1.0);
    REQUIRE((double)I(0, 1) == 0.0);
    REQUIRE((double)I(0, 2) == 0.0);
    REQUIRE((double)I(1, 0) == 0.0);
    REQUIRE((double)I(1, 1) == 1.0);
    REQUIRE((double)I(1, 2) == 0.0);
    REQUIRE((double)I(2, 0) == 0.0);
    REQUIRE((double)I(2, 1) == 0.0);
    REQUIRE((double)I(2, 2) == 1.0);
}

TEST_CASE("GPU Scale Row", "[tensor]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    DeviceTensor<double, 2> I_original{"I", einsums::detail::DEV_ONLY, 3, 3};
    I_original                     = create_random_tensor("I", 3, 3);
    DeviceTensor<double, 2> I_copy = I_original;

    scale_row(1, 2.0, &I_copy);

    REQUIRE((double)I_copy(0, 0) == (double)I_original(0, 0));
    REQUIRE((double)I_copy(0, 1) == (double)I_original(0, 1));
    REQUIRE((double)I_copy(0, 2) == (double)I_original(0, 2));
    REQUIRE((double)I_copy(1, 0) == 2.0 * (double)I_original(1, 0));
    REQUIRE((double)I_copy(1, 1) == 2.0 * (double)I_original(1, 1));
    REQUIRE((double)I_copy(1, 2) == 2.0 * (double)I_original(1, 2));
    REQUIRE((double)I_copy(2, 0) == (double)I_original(2, 0));
    REQUIRE((double)I_copy(2, 1) == (double)I_original(2, 1));
    REQUIRE((double)I_copy(2, 2) == (double)I_original(2, 2));
}

TEST_CASE("GPU Scale Column", "[tensor]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    DeviceTensor<double, 2> I_original{"I", einsums::detail::DEV_ONLY, 3, 3};
    I_original                     = create_random_tensor("I", 3, 3);
    DeviceTensor<double, 2> I_copy = I_original;

    scale_column(1, 2.0, &I_copy);

    REQUIRE((double)I_copy(0, 0) == (double)I_original(0, 0));
    REQUIRE((double)I_copy(0, 1) == 2.0 * (double)I_original(0, 1));
    REQUIRE((double)I_copy(0, 2) == (double)I_original(0, 2));
    REQUIRE((double)I_copy(1, 0) == (double)I_original(1, 0));
    REQUIRE((double)I_copy(1, 1) == 2.0 * (double)I_original(1, 1));
    REQUIRE((double)I_copy(1, 2) == (double)I_original(1, 2));
    REQUIRE((double)I_copy(2, 0) == (double)I_original(2, 0));
    REQUIRE((double)I_copy(2, 1) == 2.0 * (double)I_original(2, 1));
    REQUIRE((double)I_copy(2, 2) == (double)I_original(2, 2));
}

TEST_CASE("GPU Scale Row TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    DeviceTensor<double, 2> I_original{"I", einsums::detail::DEV_ONLY, 3, 3};
    I_original              = create_random_tensor("I", 3, 3);
    DeviceTensor     I_copy = I_original;
    DeviceTensorView I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    scale_row(1, 2.0, &I_view);

    REQUIRE((double)I_copy(0, 0) == (double)I_original(0, 0));
    REQUIRE((double)I_copy(0, 1) == (double)I_original(0, 1));
    REQUIRE((double)I_copy(0, 2) == (double)I_original(0, 2));
    REQUIRE((double)I_copy(1, 0) == (double)I_original(1, 0));
    REQUIRE((double)I_copy(1, 1) == (double)I_original(1, 1));
    REQUIRE((double)I_copy(1, 2) == (double)I_original(1, 2));
    REQUIRE((double)I_copy(2, 0) == (double)I_original(2, 0));
    REQUIRE((double)I_copy(2, 1) == 2.0 * (double)I_original(2, 1));
    REQUIRE((double)I_copy(2, 2) == 2.0 * (double)I_original(2, 2));

    REQUIRE((double)I_view(0, 0) == (double)I_original(1, 1));
    REQUIRE((double)I_view(0, 1) == (double)I_original(1, 2));
    REQUIRE((double)I_view(1, 0) == 2.0 * (double)I_original(2, 1));
    REQUIRE((double)I_view(1, 1) == 2.0 * (double)I_original(2, 2));
}

TEST_CASE("GPU Scale Column TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    DeviceTensor<double, 2> I_original{"I", einsums::detail::DEV_ONLY, 3, 3};
    I_original              = create_random_tensor("I", 3, 3);
    DeviceTensor     I_copy = I_original;
    DeviceTensorView I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    scale_column(1, 2.0, &I_view);

    REQUIRE((double)I_copy(0, 0) == (double)I_original(0, 0));
    REQUIRE((double)I_copy(0, 1) == (double)I_original(0, 1));
    REQUIRE((double)I_copy(0, 2) == (double)I_original(0, 2));
    REQUIRE((double)I_copy(1, 0) == (double)I_original(1, 0));
    REQUIRE((double)I_copy(1, 1) == (double)I_original(1, 1));
    REQUIRE((double)I_copy(1, 2) == 2.0 * (double)I_original(1, 2));
    REQUIRE((double)I_copy(2, 0) == (double)I_original(2, 0));
    REQUIRE((double)I_copy(2, 1) == (double)I_original(2, 1));
    REQUIRE((double)I_copy(2, 2) == 2.0 * (double)I_original(2, 2));

    REQUIRE((double)I_view(0, 0) == (double)I_original(1, 1));
    REQUIRE((double)I_view(0, 1) == 2.0 * (double)I_original(1, 2));
    REQUIRE((double)I_view(1, 0) == (double)I_original(2, 1));
    REQUIRE((double)I_view(1, 1) == 2.0 * (double)I_original(2, 2));
}

TEST_CASE("GPU GEMM TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    DeviceTensor<double, 2> I_original{"I", einsums::detail::DEV_ONLY, 3, 3};

    for (int i = 0, ij = 1; i < 3; i++)
        for (int j = 0; j < 3; j++, ij++)
            I_original(i, j) = ij;

    DeviceTensor     I_copy = I_original;
    DeviceTensorView I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    SECTION("Result into 2x2 matrix") {
        DeviceTensor<double, 2> result{"result", einsums::detail::DEV_ONLY, 2, 2};

        gemm<false, false>(1.0, I_view, I_view, 0.0, &result);

        REQUIRE((double)result(0, 0) == 73.0);
        REQUIRE((double)result(0, 1) == 84.0);
        REQUIRE((double)result(1, 0) == 112.0);
        REQUIRE((double)result(1, 1) == 129.0);
    }

    SECTION("Result into 2x2 view of matrix") {
        DeviceTensor<double, 2> result{"result", einsums::detail::DEV_ONLY, 5, 5};
        result.zero();
        DeviceTensorView result_view{result, Dim<2>{2, 2}, Offset<2>{3, 2}};

        gemm<false, false>(1.0, I_view, I_view, 0.0, &result_view);

        // Check view
        REQUIRE((double)result_view(0, 0) == 73.0);
        REQUIRE((double)result_view(0, 1) == 84.0);
        REQUIRE((double)result_view(1, 0) == 112.0);
        REQUIRE((double)result_view(1, 1) == 129.0);

        // Check full
        REQUIRE((double)result(3, 2) == 73.0);
        REQUIRE((double)result(3, 3) == 84.0);
        REQUIRE((double)result(4, 2) == 112.0);
        REQUIRE((double)result(4, 3) == 129.0);
    }

    SECTION("Transpose") {
        DeviceTensor<double, 2> result{"result", einsums::detail::DEV_ONLY, 2, 2};

        gemm<false, true>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE((double)result(0, 0) == 61.0);
        REQUIRE((double)result(0, 1) == 94.0);
        REQUIRE((double)result(1, 0) == 94.0);
        REQUIRE((double)result(1, 1) == 145.0);

        gemm<true, false>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE((double)result(0, 0) == 89.0);
        REQUIRE((double)result(0, 1) == 102.0);
        REQUIRE((double)result(1, 0) == 102.0);
        REQUIRE((double)result(1, 1) == 117.0);

        gemm<true, true>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE((double)result(0, 0) == 73.0);
        REQUIRE((double)result(0, 1) == 112.0);
        REQUIRE((double)result(1, 0) == 84.0);
        REQUIRE((double)result(1, 1) == 129.0);
    }
}

TEST_CASE("GPU Subset TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    SECTION("Subset View 7x7[1,:] -> 1x7") {
        const size_t size = 7;
        const size_t row  = 1;

        DeviceTensor<double, 2> I_original{"Original", einsums::detail::DEV_ONLY, size, size};
        I_original              = create_random_tensor("Original", size, size);
        DeviceTensorView I_view = I_original(row, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE((double)I_original(row, i) == (double)I_view(i));
        }
    }

    SECTION("Subset View 7x7x7[4,:,:] -> 7x7") {
        const size_t size = 7;
        const size_t d1   = 4;

        DeviceTensor<double, 3> I_original{"Original", einsums::detail::DEV_ONLY, size, size, size};
        I_original              = create_random_tensor("Original", size, size, size);
        DeviceTensorView I_view = I_original(d1, All, All);

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                REQUIRE((double)I_original(d1, i, j) == (double)I_view(i, j));
            }
        }
    }

    SECTION("Subset View 7x7x7[4,3,:] -> 7") {
        const size_t size = 7;
        const size_t d1   = 4;
        const size_t d2   = 3;

        DeviceTensor<double, 3> I_original{"Original", einsums::detail::DEV_ONLY, size, size, size};
        I_original              = create_random_tensor("Original", size, size, size);
        DeviceTensorView I_view = I_original(d1, d2, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE((double)I_original(d1, d2, i) == (double)I_view(i));
        }
    }

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> 3x3") {
        const size_t d1_size = 7, d2_size = 3, d3_size = 3;
        const size_t d1 = 4;

        DeviceTensor<double, 3> original{"Original", einsums::detail::DEV_ONLY, d1_size, d2_size, d3_size};
        original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = ij;
            }
        }

        // Obtain a 3x3 view of original[4,:,:]
        DeviceTensorView        view = original(d1, All, All);
        DeviceTensor<double, 2> result{"result", einsums::detail::DEV_ONLY, d2_size, d3_size};

        // false, false
        {
            gemm<false, false>(1.0, view, view, 0.0, &result);

            REQUIRE((double)result(0, 0) == 30.0);
            REQUIRE((double)result(0, 1) == 36.0);
            REQUIRE((double)result(0, 2) == 42.0);
            REQUIRE((double)result(1, 0) == 66.0);
            REQUIRE((double)result(1, 1) == 81.0);
            REQUIRE((double)result(1, 2) == 96.0);
            REQUIRE((double)result(2, 0) == 102.0);
            REQUIRE((double)result(2, 1) == 126.0);
            REQUIRE((double)result(2, 2) == 150.0);
        }
        // false, true
        {
            gemm<false, true>(1.0, view, view, 0.0, &result);

            REQUIRE((double)result(0, 0) == 14.0);
            REQUIRE((double)result(0, 1) == 32.0);
            REQUIRE((double)result(0, 2) == 50.0);
            REQUIRE((double)result(1, 0) == 32.0);
            REQUIRE((double)result(1, 1) == 77.0);
            REQUIRE((double)result(1, 2) == 122.0);
            REQUIRE((double)result(2, 0) == 50.0);
            REQUIRE((double)result(2, 1) == 122.0);
            REQUIRE((double)result(2, 2) == 194.0);
        }
        // true, false
        {
            gemm<true, false>(1.0, view, view, 0.0, &result);

            REQUIRE((double)result(0, 0) == 66.0);
            REQUIRE((double)result(0, 1) == 78.0);
            REQUIRE((double)result(0, 2) == 90.0);
            REQUIRE((double)result(1, 0) == 78.0);
            REQUIRE((double)result(1, 1) == 93.0);
            REQUIRE((double)result(1, 2) == 108.0);
            REQUIRE((double)result(2, 0) == 90.0);
            REQUIRE((double)result(2, 1) == 108.0);
            REQUIRE((double)result(2, 2) == 126.0);
        }
        // true, true
        {
            gemm<true, true>(1.0, view, view, 0.0, &result);

            REQUIRE((double)result(0, 0) == 30.0);
            REQUIRE((double)result(0, 1) == 66.0);
            REQUIRE((double)result(0, 2) == 102.0);
            REQUIRE((double)result(1, 0) == 36.0);
            REQUIRE((double)result(1, 1) == 81.0);
            REQUIRE((double)result(1, 2) == 126.0);
            REQUIRE((double)result(2, 0) == 42.0);
            REQUIRE((double)result(2, 1) == 96.0);
            REQUIRE((double)result(2, 2) == 150.0);
        }
    }

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> [2,:,:]") {
        // Description:
        // 1. Allocate tensor [7, 3, 3]
        // 2. Obtain view [4,:,:] (3x3 view) of tensor
        // 3. Perform GEMM and store result into view [2,:,:] (3x3 view) of tensor
        // 4. Test correctness of the GEMM result and of the data
        //    elements that should not have been touched.
        const size_t                d1_size = 7, d2_size = 3, d3_size = 3;
        const size_t                d1 = 4;
        const size_t                e1 = 2;
        const std::array<size_t, 6> untouched_d1{0, 1, 3, 4, 5, 6};

        DeviceTensor<double, 3> original{"Original", einsums::detail::DEV_ONLY, d1_size, d2_size, d3_size};
        original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = static_cast<double>(ij);
            }
        }

        DeviceTensor copy = original;

        // Obtain a 3x3 view of original[4,:,:]
        //   A view does not copy data it is just an offset pointer into the original with necessary striding information.
        DeviceTensorView view = original(d1, All, All);

        // Obtain a 3x3 view of original[2,:,:] to store the result
        DeviceTensorView result = original(e1, All, All);

        // false, false
        {
            // Call BLAS routine passing necessary offset pointer, dimension, and stride information.
            gemm<false, false>(1.0, view, view, 0.0, &result);

            // Test against the view
            REQUIRE((double)result(0, 0) == 30.0);
            REQUIRE((double)result(0, 1) == 36.0);
            REQUIRE((double)result(0, 2) == 42.0);
            REQUIRE((double)result(1, 0) == 66.0);
            REQUIRE((double)result(1, 1) == 81.0);
            REQUIRE((double)result(1, 2) == 96.0);
            REQUIRE((double)result(2, 0) == 102.0);
            REQUIRE((double)result(2, 1) == 126.0);
            REQUIRE((double)result(2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE((double)original(i, j, k) == (double)copy(i, j, k));
                    }
                }
            }
        }
        // false, true
        {
            gemm<false, true>(1.0, view, view, 0.0, &result);

            REQUIRE((double)result(0, 0) == 14.0);
            REQUIRE((double)result(0, 1) == 32.0);
            REQUIRE((double)result(0, 2) == 50.0);
            REQUIRE((double)result(1, 0) == 32.0);
            REQUIRE((double)result(1, 1) == 77.0);
            REQUIRE((double)result(1, 2) == 122.0);
            REQUIRE((double)result(2, 0) == 50.0);
            REQUIRE((double)result(2, 1) == 122.0);
            REQUIRE((double)result(2, 2) == 194.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE((double)original(i, j, k) == (double)copy(i, j, k));
                    }
                }
            }
        }
        // true, false
        {
            gemm<true, false>(1.0, view, view, 0.0, &result);

            REQUIRE((double)result(0, 0) == 66.0);
            REQUIRE((double)result(0, 1) == 78.0);
            REQUIRE((double)result(0, 2) == 90.0);
            REQUIRE((double)result(1, 0) == 78.0);
            REQUIRE((double)result(1, 1) == 93.0);
            REQUIRE((double)result(1, 2) == 108.0);
            REQUIRE((double)result(2, 0) == 90.0);
            REQUIRE((double)result(2, 1) == 108.0);
            REQUIRE((double)result(2, 2) == 126.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE((double)original(i, j, k) == (double)copy(i, j, k));
                    }
                }
            }
        }
        // true, true
        {
            gemm<true, true>(1.0, view, view, 0.0, &result);

            REQUIRE((double)result(0, 0) == 30.0);
            REQUIRE((double)result(0, 1) == 66.0);
            REQUIRE((double)result(0, 2) == 102.0);
            REQUIRE((double)result(1, 0) == 36.0);
            REQUIRE((double)result(1, 1) == 81.0);
            REQUIRE((double)result(1, 2) == 126.0);
            REQUIRE((double)result(2, 0) == 42.0);
            REQUIRE((double)result(2, 1) == 96.0);
            REQUIRE((double)result(2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE((double)original(i, j, k) == (double)copy(i, j, k));
                    }
                }
            }
        }
    }
}

TEST_CASE("GPU einsum1", "[tensor]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    SECTION("ik=ij,jk") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
                B(i, j) = ij;
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, A, Indices{index::k, index::j}, B));

        // println(A);
        // println(B);
        // println(C);

        /*[[ 30,  36,  42],
           [ 66,  81,  96],
           [102, 126, 150]]*/
        REQUIRE((double)C(0, 0) == 30.0);
        REQUIRE((double)C(0, 1) == 36.0);
        REQUIRE((double)C(0, 2) == 42.0);
        REQUIRE((double)C(1, 0) == 66.0);
        REQUIRE((double)C(1, 1) == 81.0);
        REQUIRE((double)C(1, 2) == 96.0);
        REQUIRE((double)C(2, 0) == 102.0);
        REQUIRE((double)C(2, 1) == 126.0);
        REQUIRE((double)C(2, 2) == 150.0);
    }

    SECTION("il=ijk,jkl") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 3, 3, 3};
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 3, 3, 3};
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    A(i, j, k) = ij;
                    B(i, j, k) = ij;
                }
            }
        }

        // println(A);
        // println(B);
        // println(C);

        // einsum("il=ijk,jkl", &C, A, B);
        REQUIRE_NOTHROW(
            einsum(Indices{index::i, index::l}, &C, Indices{index::i, index::j, index::k}, A, Indices{index::j, index::k, index::l}, B));

        // println(C);

        // array([[ 765.,  810.,  855.],
        //        [1818., 1944., 2070.],
        //        [2871., 3078., 3285.]])
        REQUIRE((double)C(0, 0) == 765.0);
        REQUIRE((double)C(0, 1) == 810.0);
        REQUIRE((double)C(0, 2) == 855.0);
        REQUIRE((double)C(1, 0) == 1818.0);
        REQUIRE((double)C(1, 1) == 1944.0);
        REQUIRE((double)C(1, 2) == 2070.0);
        REQUIRE((double)C(2, 0) == 2871.0);
        REQUIRE((double)C(2, 1) == 3078.0);
        REQUIRE((double)C(2, 2) == 3285.0);
    }
}

TEST_CASE("GPU einsum TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> [2,:,:]") {
        // Description: Obtain view [4,:,:] (3x3 view) perform GEMM and store result into
        // view [2,:,:] (3x3 view)
        const size_t                d1_size = 7, d2_size = 3, d3_size = 3;
        const size_t                d1 = 4;
        const size_t                e1 = 2;
        const std::array<size_t, 6> untouched1{0, 1, 3, 4, 5, 6};

        DeviceTensor<double, 3> original{"Original", einsums::detail::DEV_ONLY, d1_size, d2_size, d3_size};
        original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = static_cast<double>(ij);
            }
        }

        DeviceTensor copy = original;

        // Obtain a 3x3 view of original[4,:,:]
        DeviceTensorView view = original(d1, All, All);

        // Obtain a 3x3 view of original[2,:,:] to store the result
        DeviceTensorView result = original(e1, All, All);

        // false, false
        {
            // einsum("ik=ij,jk", &result, view, view);
            REQUIRE_NOTHROW(
                einsum(Indices{index::i, index::k}, &result, Indices{index::i, index::j}, view, Indices{index::j, index::k}, view));
            // gemm<false, false>(1.0, view, view, 0.0, &result);

            // Test against the view
            REQUIRE((double)result(0, 0) == 30.0);
            REQUIRE((double)result(0, 1) == 36.0);
            REQUIRE((double)result(0, 2) == 42.0);
            REQUIRE((double)result(1, 0) == 66.0);
            REQUIRE((double)result(1, 1) == 81.0);
            REQUIRE((double)result(1, 2) == 96.0);
            REQUIRE((double)result(2, 0) == 102.0);
            REQUIRE((double)result(2, 1) == 126.0);
            REQUIRE((double)result(2, 2) == 150.0);

            // Test the position in the original
            REQUIRE((double)original(2, 0, 0) == 30.0);
            REQUIRE((double)original(2, 0, 1) == 36.0);
            REQUIRE((double)original(2, 0, 2) == 42.0);
            REQUIRE((double)original(2, 1, 0) == 66.0);
            REQUIRE((double)original(2, 1, 1) == 81.0);
            REQUIRE((double)original(2, 1, 2) == 96.0);
            REQUIRE((double)original(2, 2, 0) == 102.0);
            REQUIRE((double)original(2, 2, 1) == 126.0);
            REQUIRE((double)original(2, 2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE((double)original(i, j, k) == (double)copy(i, j, k));
                    }
                }
            }
        }
    }
}

TEST_CASE("GPU permute2") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("Rank 2 - axpy") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        permute(Indices{i, j}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE((double)C(i, j) == (double)A(i, j));
            }
        }

        DeviceTensorView A_view{A, Dim<2>{2, 2}, Offset<2>{1, 1}};
        DeviceTensorView C_view{C, Dim<2>{2, 2}, Offset<2>{1, 1}};

        permute(Indices{j, i}, &C_view, Indices{i, j}, A_view);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                if (i == 0 || j == 0)
                    REQUIRE((double)C(i, j) == (double)A(i, j));
                else
                    CHECK((double)C(j, i) == (double)A(i, j));
            }
        }
    }

    SECTION("Rank 2 - axpy (2)") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C0{"C", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = ij + A(i, j);
            }
        }

        permute(1.0, Indices{i, j}, &C0, 1.0, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT((double)C0(i, j), Catch::Matchers::WithinRel((double)C1(i, j), 0.00001));
            }
        }

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = 2.0 * ij + 0.5 * A(i, j);
            }
        }

        permute(2.0, Indices{i, j}, &C0, 0.5, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT((double)C0(i, j), Catch::Matchers::WithinRel((double)C1(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 2") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        permute(Indices{j, i}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE_THAT((double)C(j, i), Catch::Matchers::WithinRel((double)A(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 3") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 3, 3, 3};
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        permute(Indices{k, j, i}, &B, Indices{i, j, k}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double)B(k, j, i), Catch::Matchers::WithinRel((double)A(i, j, k), 0.00001));
                }
            }
        }

        permute(Indices{i, k, j}, &B, Indices{i, j, k}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double)B(i, k, j), Catch::Matchers::WithinRel((double)A(i, j, k), 0.00001));
                }
            }
        }

        permute(Indices{j, k, i}, &B, Indices{i, j, k}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double)B(j, k, i), Catch::Matchers::WithinRel((double)A(i, j, k), 0.00001));
                }
            }
        }

        permute(Indices{i, j, k}, &B, Indices{k, j, i}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double)B(i, j, k), Catch::Matchers::WithinRel((double)A(k, j, i), 0.00001));
                }
            }
        }
    }

    SECTION("Rank 4") {
        DeviceTensor<double, 4> A{"A", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        DeviceTensor<double, 4> B{"B", einsums::detail::DEV_ONLY, 3, 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++, ij++) {
                        A(i, j, k, l) = ij;
                    }
                }
            }
        }

        permute(0.0, Indices{i, l, k, j}, &B, 0.5, Indices{k, j, l, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        REQUIRE_THAT((double)B(i, l, k, j), Catch::Matchers::WithinRel(0.5 * A(k, j, l, i), 0.00001));
                    }
                }
            }
        }
    }

    // SECTION("Rank 5") {
    //     Tensor<float, 5> A{"A", 3, 3, 3, 3, 3};
    //     Tensor<float, 5> B{"B", 3, 3, 3, 3, 3};

    //     for (short i = 0, ij = 1; i < 3; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             for (int k = 0; k < 3; k++) {
    //                 for (int l = 0; l < 3; l++) {
    //                     for (int m = 0; m < 3; m++, ij++) {
    //                         A(i, j, k, l, m) = ij;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     permute(Indices{i, k, l, m, j}, &B, Indices{j, k, l, m, i}, A);
    //     for (int i = 0; i < 3; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             for (int k = 0; k < 3; k++) {
    //                 for (int l = 0; l < 3; l++) {
    //                     for (int m = 0; m < 3; m++) {
    //                         REQUIRE(B(i, k, l, m, j) == A(j, k, l, m, i));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    SECTION("Rank 2 - Different Sizes") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 9};
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 9, 3};

        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++, ij++) {
                A(i, j) = ij;
            }
        }

        permute(Indices{j, i}, &B, Indices{i, j}, A);
        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                REQUIRE_THAT((double)B(j, i), Catch::Matchers::WithinRel((double)A(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 3 - Different Sizes") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 2, 3, 4};
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 3, 4, 2};

        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        permute(Indices{j, k, i}, &B, Indices{i, j, k}, A);
        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    REQUIRE_THAT((double)B(j, k, i), Catch::Matchers::WithinRel((double)A(i, j, k), 0.00001));
                }
            }
        }
    }
}

TEST_CASE("GPU einsum2") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("3x3 <- 3x5 * 5x3") {
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 5};
        A = create_random_tensor("A", 3, 5);
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 5, 3};
        B = create_random_tensor("B", 5, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3x3 <- 3x5 * 3x5") {
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 5};
        A = create_random_tensor("A", 3, 5);
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 3, 5};
        B = create_random_tensor("B", 3, 5);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{j, k}, B));
        linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3 <- 3x5 * 5") {
        DeviceTensor<double, 1> C0{"C0", einsums::detail::DEV_ONLY, 3};
        DeviceTensor<double, 1> C1{"C1", einsums::detail::DEV_ONLY, 3};
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 5};
        A = create_random_tensor("A", 3, 5);
        DeviceTensor<double, 1> B{"B", einsums::detail::DEV_ONLY, 5};
        B = create_random_tensor("B", 5);

        C0.zero();
        C1.zero();

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j}, A, Indices{j}, B));
        linear_algebra::gemv<false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            REQUIRE_THAT((double)C0(i0), Catch::Matchers::WithinAbs((double)C1(i0), 0.001));
        }
    }

    SECTION("3 <- 3x4x5 * 4x3x5") {
        DeviceTensor<double, 1> C0{"C0", einsums::detail::DEV_ONLY, 3};
        DeviceTensor<double, 1> C1{"C1", einsums::detail::DEV_ONLY, 3};
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 3, 4, 5};
        A = create_random_tensor("A", 3, 4, 5);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 4, 3, 5};
        B = create_random_tensor("B", 4, 3, 5);
        C0.zero();
        C1.zero();

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));

        for (size_t i0 = 0; i0 < 3; i0++) {
            double sum{0};

            for (size_t j0 = 0; j0 < 4; j0++) {
                for (size_t k0 = 0; k0 < 5; k0++) {
                    sum += (double)A(i0, j0, k0) * (double)B(j0, i0, k0);
                }
            }
            C1(i0) = sum;
        }

        for (size_t i0 = 0; i0 < 3; i0++) {
            REQUIRE_THAT((double)C0(i0), Catch::Matchers::WithinRel((double)C1(i0), 0.0001));
        }
    }

    SECTION("3x5 <- 3x4x5 * 4x3x5") {
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, 3, 5};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 5};
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 3, 4, 5};
        A = create_random_tensor("A", 3, 4, 5);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 4, 3, 5};
        B = create_random_tensor("B", 4, 3, 5);
        C0.zero();
        C1.zero();

        // profile::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, k}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // profile::pop();

        // profile::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                double sum{0};
                for (size_t j0 = 0; j0 < 4; j0++) {

                    sum += (double)A(i0, j0, k0) * (double)B(j0, i0, k0);
                }
                C1(i0, k0) = sum;
            }
        }
        // profile::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3, l <- 3x4x5 * 4x3x5") {
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, 3, 5};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 5};
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 3, 4, 5};
        A = create_random_tensor("A", 3, 4, 5);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 4, 3, 5};
        B = create_random_tensor("B", 4, 3, 5);
        C0.zero();
        C1.zero();

        // profile::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // profile::pop();

        // profile::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                for (size_t l0 = 0; l0 < 5; l0++) {
                    double sum{0};
                    for (size_t j0 = 0; j0 < 4; j0++) {

                        sum += (double)A(i0, j0, k0) * (double)B(j0, i0, k0);
                    }
                    C1(i0, l0) += sum;
                }
            }
        }
        // profile::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                // REQUIRE(C0(i0, j0) == C1(i0, j0));?
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0000001));
            }
        }
    }

    // profile::report();
    // profile::finalize();
}

TEST_CASE("GPU einsum3") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // profile::initialize();

    SECTION("3x3 <- 3x5 * 5x3") {
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 5};
        A = create_random_tensor("A", 3, 5);
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 5, 3};
        B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3x3x3x3 <- 3x3x3x3 * 3x3") {
        // This one is to represent a two-electron integral transformation
        DeviceTensor<double, 4> gMO0{"g0", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        DeviceTensor<double, 4> gMO1{"g1", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        zero(gMO0);
        zero(gMO1);
        DeviceTensor<double, 4> A{"A", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        A = create_random_tensor("A", 3, 3, 3, 3);
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 3, 3};
        B = create_random_tensor("B", 3, 3);

        auto A_host = (Tensor<double, 4>) A;
        auto B_host = (Tensor<double, 2>) B;

        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &gMO0, Indices{i, j, k, p}, A, Indices{p, l}, B));

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, k0, p0) * B(p0, l0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT((double)gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs((double)gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &gMO0, Indices{i, j, p, l}, A, Indices{p, k}, B));

        gMO1.zero();
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, p0, l0) * B(p0, k0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT((double)gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs((double)gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

        gMO0 = 2.0;
        hip_catch(hipDeviceSynchronize());
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                auto vgMO0 = gMO0(i0, j0, All, All);
                auto vA = A(i0, j0, All, All);
                einsum(Indices{k, l}, &vgMO0, Indices{p, l}, vA, Indices{p, k}, B);
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        double a = gMO0(i0, j0, k0, l0);
                        double b = gMO1(i0, j0, k0, l0);

                        if(std::abs(a - b) / std::abs(a) > 0.001) {
                            println("Failed at {} {} {} {}", i0, j0, k0, l0);
                        }
                        CHECK_THAT((double)gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs((double)gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }
    }

    // profile::report();
    // profile::finalize();
}

TEST_CASE("GPU einsum4") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // profile::initialize();
    SECTION("3x3x3x3 <- 3x3x3x3 * 3x3") {
        // This one is to represent a two-electron integral transformation
        DeviceTensor<double, 4> gMO0{"g0", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        DeviceTensor<double, 4> gMO1{"g1", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        zero(gMO0);
        zero(gMO1);
        DeviceTensor<double, 4> A{"A", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        A = create_random_tensor("A", 3, 3, 3, 3);
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 3, 3};
        B = create_random_tensor("B", 3, 3);

        REQUIRE_NOTHROW(einsum(Indices{p, q, r, l}, &gMO0, Indices{p, q, r, s}, A, Indices{s, l}, B));

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, k0, p0) * B(p0, l0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT((double)gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs((double)gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{p, q, k, s}, &gMO0, Indices{p, q, r, s}, A, Indices{r, k}, B));

        gMO1.zero();
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, p0, l0) * B(p0, k0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT((double)gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs((double)gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }
    }

    // profile::report();
    // profile::finalize();
}

TEST_CASE("GPU IntegralTransformation") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("R2 <- R3 * R3") {
        DeviceTensor<double, 2> W_mi{"W_mi", einsums::detail::DEV_ONLY, 4, 4};
        W_mi = create_random_tensor("W_mi", 4, 4);
        DeviceTensor<double, 3> g_m{"g_m", einsums::detail::DEV_ONLY, 4, 8, 8};
        g_m = create_random_tensor("g_m", 4, 8, 8);
        DeviceTensor<double, 3> t_i{"t_i", einsums::detail::DEV_ONLY, 4, 8, 8};
        t_i = create_random_tensor("t_i", 4, 8, 8);

        REQUIRE_NOTHROW(einsum(1.0, Indices{index::n, index::j}, &W_mi, 0.25, Indices{index::n, index::e, index::f}, g_m,
                               Indices{index::j, index::e, index::f}, t_i));
    }
}

TEST_CASE("GPU Hadamard") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- i,i * j*j") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, _i, _i};
        A = create_random_tensor("A", _i, _i);
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, _j, _j};
        B = create_random_tensor("B", _j, _j);
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, _i, _j};
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, _i, _j};
        C0.zero();
        C.zero();

        // println(A);
        // println(B);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, i0) * B(j0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, i}, A, Indices{j, j}, B));

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT((double)C(i0, j0), Catch::Matchers::WithinAbs((double)C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j <- i,i,j * j,j,i") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, _i, _i, _j};
        A = create_random_tensor("A", _i, _i, _j);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, _j, _j, _i};
        B = create_random_tensor("B", _j, _j, _i);
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, _i, _j};
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, _i, _j};
        C0.zero();
        C.zero();

        // println(A);
        // println(B);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, i0, j0) * B(j0, j0, i0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, i, j}, A, Indices{j, j, i}, B));

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT((double)C(i0, j0), Catch::Matchers::WithinAbs((double)C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j <- i,j,i * j,i,j") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, _i, _j, _i};
        A = create_random_tensor("A", _i, _j, _i);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, _j, _i, _j};
        B = create_random_tensor("B", _j, _i, _j);
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, _i, _j};
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, _i, _j};
        C0.zero();
        C.zero();

        // println(A);
        // println(B);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT((double)C(i0, j0), Catch::Matchers::WithinAbs((double)C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j,i <- i,j,i * j,i,j") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, _i, _j, _i};
        A = create_random_tensor("A", _i, _j, _i);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, _j, _i, _j};
        B = create_random_tensor("B", _j, _i, _j);
        DeviceTensor<double, 3> C{"C", einsums::detail::DEV_ONLY, _i, _j, _i};
        DeviceTensor<double, 3> C0{"C0", einsums::detail::DEV_ONLY, _i, _j, _i};
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CHECK_THAT((double)C(i0, j0, i0), Catch::Matchers::WithinRel((double)C0(i0, j0, i0), 0.00001));
            }
        }
    }

    SECTION("i,i,i <- i,j,i * j,i,j") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, _i, _j, _i};
        A = create_random_tensor("A", _i, _j, _i);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, _j, _i, _j};
        B = create_random_tensor("B", _j, _i, _j);
        DeviceTensor<double, 3> C{"C", einsums::detail::DEV_ONLY, _i, _i, _i};
        DeviceTensor<double, 3> C0{"C0", einsums::detail::DEV_ONLY, _i, _i, _i};
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, i0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, i, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            CHECK_THAT((double)C(i0, i0, i0), Catch::Matchers::WithinRel((double)C0(i0, i0, i0), 0.00001));
        }
    }

    SECTION("i,i <- i,j,k * j,i,k") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, _i, _j, _k};
        A = create_random_tensor("A", _i, _j, _k);
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, _j, _i, _k};
        B = create_random_tensor("B", _j, _i, _k);
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, _i, _i};
        DeviceTensor<double, 2> C0{"C0", einsums::detail::DEV_ONLY, _i, _i};
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, i0) += A(i0, j0, k0) * B(j0, i0, k0);
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, i}, &C, Indices{i, j, k}, A, Indices{j, i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _i; j0++) {
                CHECK_THAT((double)C(i0, j0), Catch::Matchers::WithinRel((double)C0(i0, j0), 0.00001));
            }
        }
    }
}

TEST_CASE("GPU unique_ptr") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("C") {
        auto                    C0 = std::make_unique<DeviceTensor<double, 2>>("C0", einsums::detail::DEV_ONLY, 3, 3);
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 5};
        A = create_random_tensor("A", 3, 5);
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 5, 3};
        B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT((double)C0->operator()(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("A") {

        DeviceTensor<double, 2> C0{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        auto                    A = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("A", 3, 5));
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 5, 3};
        B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("B") {
        DeviceTensor<double, 2> C0{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 5};
        A      = create_random_tensor("A", 3, 5);
        auto B = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("AB") {
        DeviceTensor<double, 2> C0{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        auto                    A = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("A", 3, 5));
        auto                    B = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT((double)C0(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CA") {
        auto                    C0 = std::make_unique<DeviceTensor<double, 2>>("C0", einsums::detail::DEV_ONLY, 3, 3);
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        auto                    A = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("A", 3, 5));
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 5, 3};
        B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT((double)C0->operator()(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CB") {
        auto                    C0 = std::make_unique<DeviceTensor<double, 2>>("C0", einsums::detail::DEV_ONLY, 3, 3);
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 5};
        A      = create_random_tensor("A", 3, 5);
        auto B = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT((double)C0->operator()(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CAB") {
        auto                    C0 = std::make_unique<DeviceTensor<double, 2>>("C0", einsums::detail::DEV_ONLY, 3, 3);
        DeviceTensor<double, 2> C1{"C1", einsums::detail::DEV_ONLY, 3, 3};
        auto                    A = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("A", 3, 5));
        auto                    B = std::make_unique<DeviceTensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT((double)C0->operator()(i0, j0), Catch::Matchers::WithinRel((double)C1(i0, j0), 0.0001));
            }
        }
    }
}

TEST_CASE("GPU Transpose C", "[einsum]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- j,k * k,i === true, false, false") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, _j, _k};
        A              = create_random_tensor("A", _j, _k);
        DeviceTensor B = create_random_gpu_tensor("B", _k, _i);
        DeviceTensor C{"C", _i, _j};
        DeviceTensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j, k}, A, Indices{k, i}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(j0, k0) * B(k0, i0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT((double)C(i0, j0), Catch::Matchers::WithinAbs((double)C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * k,i === true, true, false") {
        DeviceTensor A = create_random_gpu_tensor("A", _k, _j);
        DeviceTensor B = create_random_gpu_tensor("B", _k, _i);
        DeviceTensor C{"C", _i, _j};
        DeviceTensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{k, j}, A, Indices{k, i}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(k0, j0) * B(k0, i0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT((double)C(i0, j0), Catch::Matchers::WithinAbs((double)C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- j,k * i,k === true, false, true") {
        DeviceTensor A = create_random_gpu_tensor("A", _j, _k);
        DeviceTensor B = create_random_gpu_tensor("B", _i, _k);
        DeviceTensor C{"C", _i, _j};
        DeviceTensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j, k}, A, Indices{i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(j0, k0) * B(i0, k0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT((double)C(i0, j0), Catch::Matchers::WithinAbs((double)C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * i,k === true, true, true") {
        DeviceTensor A = create_random_gpu_tensor("A", _k, _j);
        DeviceTensor B = create_random_gpu_tensor("B", _i, _k);
        DeviceTensor C{"C", _i, _j};
        DeviceTensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{k, j}, A, Indices{i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(k0, j0) * B(i0, k0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT((double)C(i0, j0), Catch::Matchers::WithinAbs((double)C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("Wmnij <- 0.25 t_ijef * g_mnef") {
        size_t _m = 12, _n = 12, _i = 5, _j = 5, _e = 7, _f = 7;

        DeviceTensor Wmnij{"Wmnij", _m, _n, _i, _j};
        zero(Wmnij);
        DeviceTensor W0{"Wmnij", _m, _n, _i, _j};
        zero(W0);

        DeviceTensor t_oovv = create_random_gpu_tensor("t_oovv", _i, _j, _e, _f);
        DeviceTensor g_oovv = create_random_gpu_tensor("g_oovv", _m, _n, _e, _f);

        REQUIRE_NOTHROW(einsum(1.0, Indices{m, n, i, j}, &Wmnij, 0.25, Indices{i, j, e, f}, t_oovv, Indices{m, n, e, f}, g_oovv));

        for (size_t m0 = 0; m0 < _m; m0++) {
            for (size_t n0 = 0; n0 < _n; n0++) {
                for (size_t i0 = 0; i0 < _i; i0++) {
                    for (size_t j0 = 0; j0 < _j; j0++) {
                        for (size_t e0 = 0; e0 < _e; e0++) {
                            for (size_t f0 = 0; f0 < _f; f0++) {
                                W0(m0, n0, i0, j0) += 0.25 * t_oovv(i0, j0, e0, f0) * g_oovv(m0, n0, e0, f0);
                            }
                        }
                    }
                }
            }
        }

        for (size_t m0 = 0; m0 < _m; m0++) {
            for (size_t n0 = 0; n0 < _n; n0++) {
                for (size_t i0 = 0; i0 < _i; i0++) {
                    for (size_t j0 = 0; j0 < _j; j0++) {
                        REQUIRE_THAT((double)Wmnij(m0, n0, i0, j0), Catch::Matchers::WithinAbs((double)W0(m0, n0, i0, j0), 0.001));
                    }
                }
            }
        }
    }
}

TEST_CASE("GPU einsum_gemv") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("check") {
        size_t _p = 7, _q = 7, _r = 7, _s = 7;

        DeviceTensor g = create_random_gpu_tensor("g", _p, _q, _r, _s);
        DeviceTensor D = create_random_gpu_tensor("d", _r, _s);

        DeviceTensor F{"F", _p, _q};
        DeviceTensor F0{"F0", _p, _q};

        zero(F);
        zero(F0);

        REQUIRE_NOTHROW(einsum(1.0, Indices{p, q}, &F0, 2.0, Indices{p, q, r, s}, g, Indices{r, s}, D));

        DeviceTensorView gv{g, Dim<2>{_p * _q, _r * _s}};
        DeviceTensorView dv{D, Dim<1>{_r * _s}};
        DeviceTensorView Fv{F, Dim<1>{_p * _q}};

        linear_algebra::gemv<false>(2.0, gv, dv, 1.0, &Fv);

        // println(F0);
        // println(F);
    }
}

TEST_CASE("GPU TensorView einsum") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // Test if everything passed to einsum is a TensorView.
    DeviceTensor     A = create_random_gpu_tensor("A", 3, 5);
    DeviceTensor     B = create_random_gpu_tensor("B", 3, 5);
    DeviceTensorView A_view{A, Dim<2>{3, 3}};
    DeviceTensorView B_view{B, Dim<2>{3, 3}, Offset<2>{0, 2}};

    DeviceTensor C{"C2", 10, 10};
    C.zero();
    DeviceTensorView C_view{C, Dim<2>{3, 3}, Offset<2>{5, 5}};

    // To perform the test we make an explicit copy of the TensorViews into their own Tensors
    DeviceTensor A_copy{"A copy", 3, 3};
    DeviceTensor B_copy{"B copy", 3, 3};

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            A_copy(x, y) = A_view(x, y);
            B_copy(x, y) = B_view(x, y);
        }
    }

    // The target solution is determined from not using views
    DeviceTensor C_solution{"C solution", 3, 3};
    C_solution.zero();
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_solution, Indices{i, k}, A_copy, Indices{j, k}, B_copy));

    // einsum where everything is a TensorView
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_view, Indices{i, k}, A_view, Indices{j, k}, B_view));

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            REQUIRE_THAT((double)C_view(x, y), Catch::Matchers::WithinAbs((double)C_solution(x, y), 0.001));
            REQUIRE_THAT((double)C(x + 5, y + 5), Catch::Matchers::WithinAbs((double)C_solution(x, y), 0.001));
        }
    }
}

TEST_CASE("GPU outer product") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    size_t _x{100}, _y{100};

    SECTION("1 * 1 -> 2") {
        DeviceTensor A = create_random_gpu_tensor("A", _x);
        DeviceTensor B = create_random_gpu_tensor("B", _y);
        DeviceTensor C{"C", _x, _y};
        zero(C);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT((double)C(x, y), Catch::Matchers::WithinAbs((double)A(x) * B(y), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT((double)C(x, y), Catch::Matchers::WithinAbs((double)A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT((double)C(y, x), Catch::Matchers::WithinAbs((double)A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT((double)C(y, x), Catch::Matchers::WithinAbs((double)A(x) * B(y), 0.001));
            }
        }
    }

    SECTION("2 * 1 -> 3") {
        DeviceTensor A = create_random_gpu_tensor("A", 3, 3);
        DeviceTensor B = create_random_gpu_tensor("B", 3);
        DeviceTensor C{"C", 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT((double)C(x, y, z), Catch::Matchers::WithinAbs((double)A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT((double)C(z, x, y), Catch::Matchers::WithinAbs((double)A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{k}, B, Indices{i, j}, A));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT((double)C(z, x, y), Catch::Matchers::WithinAbs((double)A(x, y) * B(z), 0.001));
                }
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        DeviceTensor A = create_random_gpu_tensor("A", 3, 3);
        DeviceTensor B = create_random_gpu_tensor("B", 3, 3);
        DeviceTensor C{"C", 3, 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, A, Indices{k, l}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT((double)C(w, x, y, z), Catch::Matchers::WithinAbs((double)A(w, x) * B(y, z), 0.001));
                    }
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{k, l}, A, Indices{i, j}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT((double)C(w, x, y, z), Catch::Matchers::WithinAbs((double)A(y, z) * B(w, x), 0.001));
                    }
                }
            }
        }
    }
}

TEST_CASE("GPU view outer product") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("1 * 1 -> 2") {
        DeviceTensor A = create_random_gpu_tensor("A", 6);
        DeviceTensor B = create_random_gpu_tensor("B", 6);

        auto         vA = DeviceTensorView(A, Dim<1>{3}, Offset<1>{3});
        auto         vB = DeviceTensorView(B, Dim<1>{3});
        DeviceTensor C{"C", 3, 3};
        C.set_all(0.0);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, vA, Indices{j}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT((double)C(x, y), Catch::Matchers::WithinAbs((double)vA(x) * vB(y), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, vA, Indices{i}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT((double)C(x, y), Catch::Matchers::WithinAbs((double)vA(y) * vB(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, vA, Indices{i}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT((double)C(y, x), Catch::Matchers::WithinAbs((double)vA(y) * vB(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, vA, Indices{j}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT((double)C(y, x), Catch::Matchers::WithinAbs((double)vA(x) * vB(y), 0.001));
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        DeviceTensor A  = create_random_gpu_tensor("A", 9, 9);
        DeviceTensor B  = create_random_gpu_tensor("B", 12, 12);
        auto         vA = DeviceTensorView{A, Dim<2>{3, 3}, Offset<2>{6, 3}};
        auto         vB = DeviceTensorView{B, Dim<2>{3, 3}, Offset<2>{5, 7}};
        DeviceTensor C{"C", 3, 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, vA, Indices{k, l}, vB));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT((double)C(w, x, y, z), Catch::Matchers::WithinAbs((double)vA(w, x) * vB(y, z), 0.001));
                    }
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{k, l}, vA, Indices{i, j}, vB));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT((double)C(w, x, y, z), Catch::Matchers::WithinAbs((double)vA(y, z) * vB(w, x), 0.001));
                    }
                }
            }
        }
    }
}

// TEST_CASE("element transform") {
//     using namespace einsums;
//     using namespace einsums::tensor_algebra;
//     using namespace einsums::index;

//     SECTION("tensor") {
//         DeviceTensor A     = create_random_gpu_tensor("A", 32, 32, 32, 32);
//         DeviceTensor Acopy = A;

//         element_transform(&A, [](double val) -> double { return 1.0 / val; });

//         for (int w = 0; w < 3; w++) {
//             for (int x = 0; x < 3; x++) {
//                 for (int y = 0; y < 3; y++) {
//                     for (int z = 0; z < 3; z++) {
//                         REQUIRE_THAT((double) A(w, x, y, z), Catch::Matchers::WithinAbs(1.0 / (double) Acopy(w, x, y, z), 0.001));
//                     }
//                 }
//             }
//         }
//     }

//     SECTION("smartptr tensor") {
//         auto A = std::make_unique<DeviceTensor<double, 4>>("A", 32, 32, 32, 32);

//         element_transform(&A, [](double val) -> double { return 1.0 / val; });
//     }
// }

// TEST_CASE("element") {
//     using namespace einsums;
//     using namespace einsums::tensor_algebra;
//     using namespace einsums::index;

//     SECTION("1") {
//         Tensor A     = create_random_tensor("A", 10, 10, 10, 10);
//         Tensor Acopy = A;

//         Tensor B = create_random_tensor("B", 10, 10, 10, 10);

//         element([](double const &Aval, double const &Bval) -> double { return Aval + Bval; }, &A, B);

//         for (int w = 0; w < 10; w++) {
//             for (int x = 0; x < 10; x++) {
//                 for (int y = 0; y < 10; y++) {
//                     for (int z = 0; z < 10; z++) {
//                         REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(Acopy(w, x, y, z) + B(w, x, y, z), 0.001));
//                     }
//                 }
//             }
//         }
//     }

//     SECTION("2") {
//         Tensor A     = create_random_tensor("A", 10, 10, 10, 10);
//         Tensor Acopy = A;

//         Tensor B = create_random_tensor("B", 10, 10, 10, 10);
//         Tensor C = create_random_tensor("C", 10, 10, 10, 10);

//         element([](double const &Aval, double const &Bval, double const &Cval) -> double { return Aval + Bval + Cval; }, &A, B, C);

//         for (int w = 0; w < 10; w++) {
//             for (int x = 0; x < 10; x++) {
//                 for (int y = 0; y < 10; y++) {
//                     for (int z = 0; z < 10; z++) {
//                         REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(Acopy(w, x, y, z) + B(w, x, y, z) + C(w, x, y, z),
//                         0.001));
//                     }
//                 }
//             }
//         }
//     }

//     // SECTION("2 - error") {

//     //     Tensor A = create_random_tensor("A", 10, 10, 10, 10);
//     //     Tensor Acopy = A;

//     //     Tensor B = create_random_tensor("B", 10, 10, 10, 9);
//     //     Tensor C = create_random_tensor("C", 10, 10, 10, 10);

//     //     element(
//     //         &A, [](double const &Aval, double const &Bval, double const &Cval) -> double { return Aval + Bval + Cval; }, B, C);
//     // }
// }

// TEST_CASE("einsum element") {
//     using namespace einsums;
//     using namespace einsums::tensor_algebra;
//     using namespace einsums::index;

//     const int _i{50}, _j{50};

//     SECTION("1") {
//         Tensor C  = Tensor{"C", _i, _j};
//         Tensor C0 = Tensor{"C", _i, _j};

//         Tensor B = create_random_tensor("B", _i, _j);
//         Tensor A = create_random_tensor("A", _i, _j);

//         element([](double const & /*Cval*/, double const &Aval, double const &Bval) -> double { return Aval * Bval; }, &C0, A, B);

//         einsum(Indices{i, j}, &C, Indices{i, j}, A, Indices{i, j}, B);

//         for (int w = 0; w < _i; w++) {
//             for (int x = 0; x < _j; x++) {
//                 REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(C0(w, x), 1.0e-5));
//             }
//         }
//     }

//     SECTION("2") {
//         Tensor C  = create_random_tensor("C", _i, _j);
//         Tensor C0 = C;
//         Tensor testresult{"result", _i, _j};
//         zero(testresult);

//         Tensor A = create_random_tensor("A", _i, _j);

//         element([](double const &Cval, double const &Aval) -> double { return Cval * Aval; }, &C, A);

//         einsum(Indices{i, j}, &testresult, Indices{i, j}, C0, Indices{i, j}, A);

//         for (int w = 0; w < _i; w++) {
//             for (int x = 0; x < _j; x++) {
//                 REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(testresult(w, x), 1.0e-5));
//             }
//         }
//     }

//     SECTION("3") {
//         Tensor parentC  = create_random_tensor("parentC", _i, _i, _i, _j);
//         Tensor parentC0 = parentC;
//         Tensor parentA  = create_random_tensor("parentA", _i, _i, _i, _j);

//         auto   C  = parentC(3, All, All, 4);
//         auto   C0 = parentC0(3, All, All, 4);
//         Tensor testresult{"result", _i, _j};

//         for (int w = 0; w < _i; w++) {
//             for (int x = 0; x < _j; x++) {
//                 testresult(w, x) = C(w, x);
//             }
//         }

//         auto A = parentA(1, 2, All, All);

//         element([](double const &Cval, double const &Aval) -> double { return Cval * Aval; }, &C, A);

//         einsum(Indices{i, j}, &testresult, Indices{i, j}, C0, Indices{i, j}, A);

//         for (int w = 0; w < _i; w++) {
//             for (int x = 0; x < _j; x++) {
//                 REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(testresult(w, x), 1.0e-5));
//             }
//         }
//     }
// }

TEST_CASE("GPU F12 - V term") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // int nocc{5}, ncabs{116}, nobs{41};
    int          nocc{1}, ncabs{4}, nobs{2};
    int          nall{nobs + ncabs};
    DeviceTensor F{"F", nall, nall, nall, nall};
    DeviceTensor G{"G", nall, nall, nall, nall};

    F = create_incremented_tensor("F", nall, nall, nall, nall);
    G = create_incremented_tensor("G", nall, nall, nall, nall);

    DeviceTensorView F_ooco{F, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    DeviceTensorView F_oooc{F, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    DeviceTensorView F_oopq{F, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};
    DeviceTensorView G_ooco{G, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    DeviceTensorView G_oooc{G, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    DeviceTensorView G_oopq{G, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};

    DeviceTensor ijkl_1 = DeviceTensor{"Einsum Temp 1", nocc, nocc, nocc, nocc};
    DeviceTensor ijkl_2 = DeviceTensor{"Einsum Temp 2", nocc, nocc, nocc, nocc};
    DeviceTensor ijkl_3 = DeviceTensor{"Einsum Temp 3", nocc, nocc, nocc, nocc};

    ijkl_1.set_all(0.0);
    ijkl_2.set_all(0.0);
    ijkl_3.set_all(0.0);

    DeviceTensor result  = DeviceTensor{"Result", nocc, nocc, nocc, nocc};
    DeviceTensor result2 = DeviceTensor{"Result2", nocc, nocc, nocc, nocc};

    // println(F);
    // println(G);

    einsum(Indices{i, j, k, l}, &ijkl_1, Indices{i, j, p, n}, G_ooco, Indices{k, l, p, n}, F_ooco);
    einsum(Indices{i, j, k, l}, &ijkl_2, Indices{i, j, m, q}, G_oooc, Indices{k, l, m, q}, F_oooc);
    einsum(Indices{i, j, k, l}, &ijkl_3, Indices{i, j, p, q}, G_oopq, Indices{k, l, p, q}, F_oopq);

    gpu::device_synchronize();

    result.set_all(0.0);
    result2.set_all(0.0);
    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    for (size_t _p = 0; _p < ncabs; _p++) {
                        for (size_t _n = 0; _n < nocc; _n++) {
                            // println("A({}, {}, {}, {}) = {}", _i, _j, _p, _n, G_ooco(_i, _j, _p, _n));
                            // println("B({}, {}, {}, {}) = {}", _k, _l, _p, _n, F_ooco(_k, _l, _p, _n));

                            result(_i, _j, _k, _l) += G(_i, _j, nobs + _p, _n) * F(_k, _l, nobs + _p, _n);
                            result2(_i, _j, _k, _l) += G_ooco(_i, _j, _p, _n) * F_ooco(_k, _l, _p, _n);
                        }
                    }
                }
            }
        }
    }

    // println(result);
    // println(ijkl_1);

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT((double)result2(_i, _j, _k, _l), Catch::Matchers::WithinAbs((double)result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT((double)ijkl_1(_i, _j, _k, _l), Catch::Matchers::WithinAbs((double)result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }
}

TEST_CASE("GPU B_tilde") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{5}, ncabs{10}, nobs{10};
    assert(nobs > nocc); // sanity check
    int nall{nobs + ncabs}, nvir{nobs - nocc};

    DeviceTensor CD{"CD", nocc, nocc, nvir, nvir};
    DeviceTensor CD0{"CD0", nocc, nocc, nvir, nvir};
    zero(CD);
    zero(CD0);
    auto C    = create_random_gpu_tensor("C", nocc, nocc, nvir, nvir);
    auto D    = create_random_gpu_tensor("D", nocc, nocc, nvir, nvir);
    auto D_ij = D(2, 2, All, All);

    einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, C, Indices{a, b}, D_ij);

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    CD0(_k, _l, _a, _b) = C(_k, _l, _a, _b) * D(2, 2, _a, _b);
                }
            }
        }
    }

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    REQUIRE_THAT((double)CD(_k, _l, _a, _b), Catch::Matchers::WithinAbs((double)CD0(_k, _l, _a, _b), 0.000001));
                }
            }
        }
    }
}

TEST_CASE("GPU Khatri-Rao") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    const int _I{8}, _M{4}, _r{16};

    SECTION("einsum") {

        auto KR  = DeviceTensor{"KR", _I, _M, _r};
        auto KR0 = DeviceTensor{"KR0", _I, _M, _r};

        auto T = create_random_gpu_tensor("T", _I, _r);
        auto U = create_random_gpu_tensor("U", _M, _r);

        einsum(Indices{I, M, r}, &KR, Indices{I, r}, T, Indices{M, r}, U);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    REQUIRE_THAT((double)KR(x, y, z), Catch::Matchers::WithinAbs((double)KR0(x, y, z), 0.000001));
                }
            }
        }
    }

    SECTION("special function") {
        auto KR0 = DeviceTensor{"KR0", _I, _M, _r};

        auto T = create_random_gpu_tensor("T", _I, _r);
        auto U = create_random_gpu_tensor("U", _M, _r);

        auto KR = khatri_rao(Indices{I, r}, T, Indices{M, r}, U);
        // println(result);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        auto KR0_view = DeviceTensorView{KR0, Dim<2>{_I * _M, _r}};

        for (int x = 0; x < _I * _M; x++) {
            for (int z = 0; z < _r; z++) {
                REQUIRE_THAT((double)KR(x, z), Catch::Matchers::WithinAbs((double)KR0_view(x, z), 0.000001));
            }
        }
    }
}

template <typename TC, typename TA, typename TB>
void einsum_gpu_mixed_test() {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    const auto i_ = 10, j_ = 10, k_ = 10;

    auto A  = create_random_gpu_tensor<TA>("A", i_, k_);
    auto B  = create_random_gpu_tensor<TB>("B", k_, j_);
    auto C  = DeviceTensor<TC, 2>("C", i_, j_);
    auto C0 = DeviceTensor<TC, 2>("C0", i_, j_);
    zero(C);
    zero(C0);

    einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t i = 0; i < i_; i++) {
        for (size_t j = 0; j < j_; j++) {
            for (size_t k = 0; k < k_; k++) {
                C0(i, j) += (TA)A(i, k) * (TB)B(k, j);
            }
        }
    }

    for (size_t i = 0; i < i_; i++) {
        for (size_t j = 0; j < j_; j++) {
            // println("{:20.14f} {:20.14f} {:20.14f}", C(i, j), C0(i, j), std::abs(C(i, j) - C0(i, j)));
            CHECK(std::abs((TC)C(i, j) - (TC)C0(i, j)) < RemoveComplexT<TC>{1.0E-4});
            // REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(C0(i, j), RemoveComplexT<TC>{1.0E-16}));
        }
    }
}

TEST_CASE("GPU einsum-mixed") {
    SECTION("d-f-d") {
        einsum_gpu_mixed_test<double, float, double>();
    }
    SECTION("d-f-f") {
        einsum_gpu_mixed_test<double, float, float>();
    }
    SECTION("f-d-d") {
        einsum_gpu_mixed_test<float, double, double>();
    }
    SECTION("cd-cd-d") {
        einsum_gpu_mixed_test<std::complex<double>, std::complex<double>, double>();
    }
    SECTION("d-d-d") {
        einsum_gpu_mixed_test<double, double, double>();
    }
    // VERY SENSITIVE
    // SECTION("cf-cd-f") {
    //     einsum_mixed_test<std::complex<float>, std::complex<float>, std::complex<float>>();
    // }
}

template <typename T>
void dot_test() {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    using namespace einsums::linear_algebra;

    size_t i_{10}, j_{10}, a_{10}, b_{10};

    RemoveComplexT<T> error;

    if constexpr (std::is_same_v<RemoveComplexT<T>, float>) {
        error = 0.001;
    } else {
        error = 0.00001;
    }

    SECTION("1") {
        auto               A = create_random_gpu_tensor<T>("A", i_);
        auto               B = create_random_gpu_tensor<T>("B", i_);
        DeviceTensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i}, A, Indices{i}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT((T)C_obtained, Catch::Matchers::WithinAbsMatcher((T)C_expected, error));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), error));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), error));
        }
    }

    SECTION("2") {
        auto               A = create_random_gpu_tensor<T>("A", i_, j_);
        auto               B = create_random_gpu_tensor<T>("B", i_, j_);
        DeviceTensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j}, A, Indices{i, j}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT((T)C_obtained, Catch::Matchers::WithinAbsMatcher((T)C_expected, error));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), error));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), error));
        }
    }

    SECTION("3") {
        auto               A = create_random_gpu_tensor<T>("A", i_, j_, a_);
        auto               B = create_random_gpu_tensor<T>("B", i_, j_, a_);
        DeviceTensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a}, A, Indices{i, j, a}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT((T)C_obtained, Catch::Matchers::WithinAbsMatcher((T)C_expected, error));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), error));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), error));
        }
    }

    SECTION("4") {
        auto               A = create_random_gpu_tensor<T>("A", i_, j_, a_, b_);
        auto               B = create_random_gpu_tensor<T>("B", i_, j_, a_, b_);
        DeviceTensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a, b}, A, Indices{i, j, a, b}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT((T)C_obtained, Catch::Matchers::WithinAbsMatcher((T)C_expected, error));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), error));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), error));
        }
    }
}

TEST_CASE("GPU dot") {
    SECTION("float") {
        dot_test<float>();
    }
    SECTION("double") {
        dot_test<double>();
    }
    SECTION("cfloat") {
        dot_test<std::complex<float>>();
    }
    SECTION("cdouble") {
        dot_test<std::complex<double>>();
    }
}

TEST_CASE("GPU Dot TensorView and Tensor") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    const auto i_ = 10, j_ = 10, k_ = 10, l_ = 2;

    auto A  = create_random_gpu_tensor<double>("A", i_, k_);
    auto B  = create_random_gpu_tensor<double>("B", k_, j_);
    auto C  = DeviceTensor<double, 2>("C", l_, j_);
    auto C0 = DeviceTensor<double, 2>("C0", l_, j_);
    zero(C0);

    auto A_view = A(Range{0, l_}, All); // (l_, k_)

    einsum(Indices{l, j}, &C, Indices{l, k}, A_view, Indices{k, j}, B);

    for (size_t l = 0; l < l_; l++) {
        for (size_t j = 0; j < j_; j++) {
            for (size_t k = 0; k < k_; k++) {
                C0(l, j) += A(l, k) * B(k, j);
            }
        }
    }

    for (size_t l = 0; l < l_; l++) {
        for (size_t j = 0; j < j_; j++) {
            // println("{:20.14f} {:20.14f} {:20.14f}", C(l, j), C0(l, j), std::abs(C(l, j) - C0(l, j)));
            REQUIRE_THAT((double)C(l, j), Catch::Matchers::WithinAbs((double)C0(l, j), 1e-12));
        }
    }
}

TEST_CASE("GPU andy") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    size_t proj_rank_{10}, nocc_{5}, nvirt_{28}, naux_{3}, u_rank_{4};

    SECTION("1") {
        auto y_iW_        = create_random_gpu_tensor("y_iW", nocc_, proj_rank_);
        auto y_aW_        = create_random_gpu_tensor("y_aW", nvirt_, proj_rank_);
        auto ortho_temp_1 = DeviceTensor("ortho temp 1", nocc_, nvirt_, proj_rank_);

        zero(ortho_temp_1);
        einsum(0.0, Indices{index::i, index::a, index::W}, &ortho_temp_1, 1.0, Indices{index::i, index::W}, y_iW_,
               Indices{index::a, index::W}, y_aW_);
    }

    SECTION("2") {
        auto tau_         = create_random_gpu_tensor("tau", proj_rank_, proj_rank_);
        auto ortho_temp_1 = create_random_gpu_tensor("ortho temp 1", nocc_, nvirt_, proj_rank_);
        auto ortho_temp_2 = DeviceTensor("ortho temp 2", nocc_, nvirt_, proj_rank_);

        zero(ortho_temp_2);
        einsum(0.0, Indices{index::i, index::a, index::P}, &ortho_temp_2, 1.0, Indices{index::i, index::a, index::W}, ortho_temp_1,
               Indices{index::P, index::W}, tau_);
    }

    SECTION("3") {
        auto a = create_random_gpu_tensor("a", nvirt_, nvirt_);
        auto b = create_random_gpu_tensor("b", nvirt_, nvirt_);
        auto c = DeviceTensor("c", nvirt_, nvirt_);
        zero(c);

        einsum(0.0, Indices{index::p, index::q}, &c, -1.0, Indices{index::p, index::q}, a, Indices{index::p, index::q}, b);
        gpu::device_synchronize();

        for (int x = 0; x < nvirt_; x++) {
            for (int y = 0; y < nvirt_; y++) {
                REQUIRE_THAT((double)c(x, y), Catch::Matchers::WithinRel((double)-a(x, y) * b(x, y)));
            }
        }
    }

    SECTION("4") {
        auto A  = create_random_gpu_tensor("a", proj_rank_, nocc_, nvirt_);
        auto B  = create_random_gpu_tensor("b", nocc_, nvirt_, proj_rank_);
        auto c  = DeviceTensor("c", proj_rank_, proj_rank_);
        auto c0 = DeviceTensor("c0", proj_rank_, proj_rank_);

        zero(c);
        einsum(Indices{index::Q, index::X}, &c, Indices{index::Q, index::i, index::a}, A, Indices{index::i, index::a, index::X}, B);

        zero(c0);
        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t X = 0; X < proj_rank_; X++) {
                for (size_t i = 0; i < nocc_; i++) {
                    for (size_t a = 0; a < nvirt_; a++) {
                        c0(Q, X) += A(Q, i, a) * B(i, a, X);
                    }
                }
            }
        }

        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t X = 0; X < proj_rank_; X++) {
                REQUIRE_THAT((double)c(Q, X), Catch::Matchers::WithinRel((double)c0(Q, X), 0.00001));
            }
        }
    }

    SECTION("5") {
        auto F_TEMP = create_random_gpu_tensor("F_TEMP", proj_rank_, proj_rank_, proj_rank_);
        auto y_aW   = create_random_gpu_tensor("y_aW", nvirt_, proj_rank_);
        auto F_BAR  = DeviceTensor("F_BAR", proj_rank_, nvirt_, proj_rank_);
        auto F_BAR0 = DeviceTensor("F_BAR", proj_rank_, nvirt_, proj_rank_);

        zero(F_BAR);
        einsum(Indices{index::Q, index::a, index::X}, &F_BAR, Indices{index::Q, index::Y, index::X}, F_TEMP, Indices{index::a, index::Y},
               y_aW);

        zero(F_BAR0);
        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t a = 0; a < nvirt_; a++) {
                for (size_t X = 0; X < proj_rank_; X++) {
                    for (size_t Y = 0; Y < proj_rank_; Y++) {
                        F_BAR0(Q, a, X) += F_TEMP(Q, Y, X) * y_aW(a, Y);
                    }
                }
            }
        }

        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t a = 0; a < nvirt_; a++) {
                for (size_t X = 0; X < proj_rank_; X++) {
                    REQUIRE_THAT((double)F_BAR(Q, a, X), Catch::Matchers::WithinRel((double)F_BAR0(Q, a, X), 0.00001));
                }
            }
        }
    }

    SECTION("6") {
        auto A = create_random_gpu_tensor("A", 84);
        auto C = DeviceTensor("C", 84, 84);
        zero(C);

        einsum(Indices{index::a, index::b}, &C, Indices{index::a}, A, Indices{index::b}, A);

        for (size_t a = 0; a < 84; a++) {
            for (size_t b = 0; b < 84; b++) {
                REQUIRE_THAT((double)C(a, b), Catch::Matchers::WithinRel((double)A(a) * A(b), 0.00001));
            }
        }
    }

    SECTION("7") {
        auto A = DeviceTensor("A", 9);
        A(0)   = 0.26052754;
        A(1)   = 0.20708203;
        A(2)   = 0.18034861;
        A(3)   = 0.18034861;
        A(4)   = 0.10959806;
        A(5)   = 0.10285149;
        A(6)   = 0.10285149;
        A(7)   = 0.10164104;
        A(8)   = 0.06130642;
        auto C = DeviceTensor("C", 9, 9);
        zero(C);

        einsum(Indices{index::a, index::b}, &C, Indices{index::a}, A, Indices{index::b}, A);

        for (size_t a = 0; a < 9; a++) {
            for (size_t b = 0; b < 9; b++) {
                REQUIRE_THAT((double)C(a, b), Catch::Matchers::WithinRel((double)A(a) * A(b), 0.00001));
            }
        }
    }

    SECTION("8") {
        auto C_TILDE = create_random_gpu_tensor("C_TILDE", naux_, nvirt_, u_rank_);
        auto B_QY    = create_random_gpu_tensor("B_QY", naux_, u_rank_);

        auto D_TILDE = DeviceTensor("D_TILDE", nvirt_, u_rank_);
        zero(D_TILDE);

        einsum(0.0, Indices{index::a, index::X}, &D_TILDE, 1.0, Indices{index::Q, index::a, index::X}, C_TILDE, Indices{index::Q, index::X},
               B_QY);
    }

    SECTION("9") {
        auto Qov  = create_random_gpu_tensor("Qov", naux_, nocc_, nvirt_);
        auto ia_X = create_random_gpu_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto N_QX = DeviceTensor("N_QX", naux_, u_rank_);
        zero(N_QX);

        einsum(Indices{index::Q, index::X}, &N_QX, Indices{index::Q, index::i, index::a}, Qov, Indices{index::i, index::a, index::X}, ia_X);
    }

    SECTION("10") {
        auto t_ia = create_random_gpu_tensor("t_ia", nocc_, nvirt_);
        auto ia_X = create_random_gpu_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto M_X = DeviceTensor("M_X", u_rank_);
        zero(M_X);

        einsum(Indices{index::X}, &M_X, Indices{index::i, index::a, index::X}, ia_X, Indices{index::i, index::a}, t_ia);
    }

    SECTION("11") {
        auto B_Qmo = create_random_gpu_tensor("Q", naux_, nocc_ + nvirt_, nocc_ + nvirt_);
        // println(B_Qmo);
        auto Qov = B_Qmo(All, Range{0, nocc_}, Range{nocc_, nocc_ + nvirt_});

        // println(Qov, {.full_output = false});

        auto ia_X = create_random_gpu_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto N_QX = DeviceTensor("N_QX", naux_, u_rank_);
        zero(N_QX);

        einsum(Indices{index::Q, index::X}, &N_QX, Indices{index::Q, index::i, index::a}, Qov, Indices{index::i, index::a, index::X}, ia_X);
    }
}