#include "einsums/TensorAlgebra.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/State.hpp"
#include "einsums/Tensor.hpp"

#include <H5Fpublic.h>
#include <catch2/catch.hpp>
#include <type_traits>

TEST_CASE("Identity Tensor", "[tensor]") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;

    Tensor<2, double> I = create_identity_tensor("I", 3, 3);

    REQUIRE(I(0, 0) == 1.0);
    REQUIRE(I(0, 1) == 0.0);
    REQUIRE(I(0, 2) == 0.0);
    REQUIRE(I(1, 0) == 0.0);
    REQUIRE(I(1, 1) == 1.0);
    REQUIRE(I(1, 2) == 0.0);
    REQUIRE(I(2, 0) == 0.0);
    REQUIRE(I(2, 1) == 0.0);
    REQUIRE(I(2, 2) == 1.0);
}

TEST_CASE("Scale Row", "[tensor]") {
    using namespace einsums;
    using namespace einsums::LinearAlgebra;

    Tensor<2> I_original = create_random_tensor("I", 3, 3);
    Tensor<2> I_copy = I_original;

    scale_row(1, 2.0, &I_copy);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == 2.0 * I_original(1, 0));
    REQUIRE(I_copy(1, 1) == 2.0 * I_original(1, 1));
    REQUIRE(I_copy(1, 2) == 2.0 * I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == I_original(2, 1));
    REQUIRE(I_copy(2, 2) == I_original(2, 2));
}

TEST_CASE("Scale Column", "[tensor]") {
    using namespace einsums;
    using namespace einsums::LinearAlgebra;

    Tensor<2> I_original = create_random_tensor("I", 3, 3);
    Tensor<2> I_copy = I_original;

    scale_column(1, 2.0, &I_copy);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == 2.0 * I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == I_original(1, 0));
    REQUIRE(I_copy(1, 1) == 2.0 * I_original(1, 1));
    REQUIRE(I_copy(1, 2) == I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == 2.0 * I_original(2, 1));
    REQUIRE(I_copy(2, 2) == I_original(2, 2));
}

TEST_CASE("Scale Row TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::LinearAlgebra;

    Tensor<2> I_original = create_random_tensor("I", 3, 3);
    Tensor<2> I_copy = I_original;
    TensorView<2> I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    scale_row(1, 2.0, &I_view);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == I_original(1, 0));
    REQUIRE(I_copy(1, 1) == I_original(1, 1));
    REQUIRE(I_copy(1, 2) == I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == 2.0 * I_original(2, 1));
    REQUIRE(I_copy(2, 2) == 2.0 * I_original(2, 2));

    REQUIRE(I_view(0, 0) == I_original(1, 1));
    REQUIRE(I_view(0, 1) == I_original(1, 2));
    REQUIRE(I_view(1, 0) == 2.0 * I_original(2, 1));
    REQUIRE(I_view(1, 1) == 2.0 * I_original(2, 2));
}

TEST_CASE("Scale Column TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::LinearAlgebra;

    Tensor<2> I_original = create_random_tensor("I", 3, 3);
    Tensor<2> I_copy = I_original;
    TensorView<2> I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    scale_column(1, 2.0, &I_view);

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == I_original(1, 0));
    REQUIRE(I_copy(1, 1) == I_original(1, 1));
    REQUIRE(I_copy(1, 2) == 2.0 * I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == I_original(2, 1));
    REQUIRE(I_copy(2, 2) == 2.0 * I_original(2, 2));

    REQUIRE(I_view(0, 0) == I_original(1, 1));
    REQUIRE(I_view(0, 1) == 2.0 * I_original(1, 2));
    REQUIRE(I_view(1, 0) == I_original(2, 1));
    REQUIRE(I_view(1, 1) == 2.0 * I_original(2, 2));
}

TEST_CASE("GEMM TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::LinearAlgebra;

    Tensor<2> I_original{"I", 3, 3};

    for (int i = 0, ij = 1; i < 3; i++)
        for (int j = 0; j < 3; j++, ij++)
            I_original(i, j) = ij;

    Tensor<2> I_copy = I_original;
    TensorView<2> I_view{I_copy, Dim<2>{2, 2}, Offset<2>{1, 1}};

    SECTION("Result into 2x2 matrix") {
        Tensor<2> result{"result", 2, 2};

        gemm<false, false>(1.0, I_view, I_view, 0.0, &result);

        REQUIRE(result(0, 0) == 73.0);
        REQUIRE(result(0, 1) == 84.0);
        REQUIRE(result(1, 0) == 112.0);
        REQUIRE(result(1, 1) == 129.0);
    }

    SECTION("Result into 2x2 view of matrix") {
        Tensor<2> result{"result", 5, 5};
        TensorView<2> result_view{result, Dim<2>{2, 2}, Offset<2>{3, 2}};

        gemm<false, false>(1.0, I_view, I_view, 0.0, &result_view);

        // Check view
        REQUIRE(result_view(0, 0) == 73.0);
        REQUIRE(result_view(0, 1) == 84.0);
        REQUIRE(result_view(1, 0) == 112.0);
        REQUIRE(result_view(1, 1) == 129.0);

        // Check full
        REQUIRE(result(3, 2) == 73.0);
        REQUIRE(result(3, 3) == 84.0);
        REQUIRE(result(4, 2) == 112.0);
        REQUIRE(result(4, 3) == 129.0);
    }

    SECTION("Transpose") {
        Tensor<2> result{"result", 2, 2};

        gemm<false, true>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE(result(0, 0) == 61.0);
        REQUIRE(result(0, 1) == 94.0);
        REQUIRE(result(1, 0) == 94.0);
        REQUIRE(result(1, 1) == 145.0);

        gemm<true, false>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE(result(0, 0) == 89.0);
        REQUIRE(result(0, 1) == 102.0);
        REQUIRE(result(1, 0) == 102.0);
        REQUIRE(result(1, 1) == 117.0);

        gemm<true, true>(1.0, I_view, I_view, 0.0, &result);
        REQUIRE(result(0, 0) == 73.0);
        REQUIRE(result(0, 1) == 112.0);
        REQUIRE(result(1, 0) == 84.0);
        REQUIRE(result(1, 1) == 129.0);
    }
}

TEST_CASE("Subset TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::LinearAlgebra;

    SECTION("Subset View 7x7[1,:] -> 1x7") {
        const size_t size = 7;
        const size_t row = 1;

        Tensor<2> I_original = create_random_tensor("Original", size, size);
        TensorView<1> I_view = I_original(row, All{});

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(row, i) == I_view(i));
        }
    }

    SECTION("Subset View 7x7x7[4,:,:] -> 7x7") {
        const size_t size = 7;
        const size_t d1 = 4;

        Tensor<3> I_original = create_random_tensor("Original", size, size, size);
        TensorView<2> I_view = I_original(d1, All{}, All{});

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                REQUIRE(I_original(d1, i, j) == I_view(i, j));
            }
        }
    }

    SECTION("Subset View 7x7x7[4,3,:] -> 7") {
        const size_t size = 7;
        const size_t d1 = 4;
        const size_t d2 = 3;

        Tensor<3> I_original = create_random_tensor("Original", size, size, size);
        TensorView<1> I_view = I_original(d1, d2, All{});

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(d1, d2, i) == I_view(i));
        }
    }

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> 3x3") {
        const size_t d1_size = 7, d2_size = 3, d3_size = 3;
        const size_t d1 = 4;

        Tensor<3> original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = ij;
            }
        }

        // Obtain a 3x3 view of original[4,:,:]
        TensorView<2> view = original(d1, All{}, All{});
        Tensor<2> result{"result", d2_size, d3_size};

        // false, false
        {
            gemm<false, false>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 36.0);
            REQUIRE(result(0, 2) == 42.0);
            REQUIRE(result(1, 0) == 66.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 96.0);
            REQUIRE(result(2, 0) == 102.0);
            REQUIRE(result(2, 1) == 126.0);
            REQUIRE(result(2, 2) == 150.0);
        }
        // false, true
        {
            gemm<false, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 14.0);
            REQUIRE(result(0, 1) == 32.0);
            REQUIRE(result(0, 2) == 50.0);
            REQUIRE(result(1, 0) == 32.0);
            REQUIRE(result(1, 1) == 77.0);
            REQUIRE(result(1, 2) == 122.0);
            REQUIRE(result(2, 0) == 50.0);
            REQUIRE(result(2, 1) == 122.0);
            REQUIRE(result(2, 2) == 194.0);
        }
        // true, false
        {
            gemm<true, false>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 66.0);
            REQUIRE(result(0, 1) == 78.0);
            REQUIRE(result(0, 2) == 90.0);
            REQUIRE(result(1, 0) == 78.0);
            REQUIRE(result(1, 1) == 93.0);
            REQUIRE(result(1, 2) == 108.0);
            REQUIRE(result(2, 0) == 90.0);
            REQUIRE(result(2, 1) == 108.0);
            REQUIRE(result(2, 2) == 126.0);
        }
        // true, true
        {
            gemm<true, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 66.0);
            REQUIRE(result(0, 2) == 102.0);
            REQUIRE(result(1, 0) == 36.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 126.0);
            REQUIRE(result(2, 0) == 42.0);
            REQUIRE(result(2, 1) == 96.0);
            REQUIRE(result(2, 2) == 150.0);
        }
    }

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> [2,:,:]") {
        // Description:
        // 1. Allocate tensor [7, 3, 3]
        // 2. Obtain view [4,:,:] (3x3 view) of tensor
        // 3. Perform GEMM and store result into view [2,:,:] (3x3 view) of tensor
        // 4. Test correctness of the GEMM result and of the data
        //    elements that should not have been touched.
        const size_t d1_size = 7, d2_size = 3, d3_size = 3;
        const size_t d1 = 4;
        const size_t e1 = 2;
        const std::array<size_t, 6> untouched_d1{0, 1, 3, 4, 5, 6};

        Tensor<3> original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = ij;
            }
        }

        Tensor<3> copy = original;

        // Obtain a 3x3 view of original[4,:,:]
        //   A view does not copy data it is just an offset pointer into the original with necessary striding information.
        TensorView<2> view = original(d1, All{}, All{});

        // Obtain a 3x3 view of original[2,:,:] to store the result
        TensorView<2> result = original(e1, All{}, All{});

        // false, false
        {
            // Call BLAS routine passing necessary offset pointer, dimension, and stride information.
            gemm<false, false>(1.0, view, view, 0.0, &result);

            // Test against the view
            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 36.0);
            REQUIRE(result(0, 2) == 42.0);
            REQUIRE(result(1, 0) == 66.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 96.0);
            REQUIRE(result(2, 0) == 102.0);
            REQUIRE(result(2, 1) == 126.0);
            REQUIRE(result(2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
        // false, true
        {
            gemm<false, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 14.0);
            REQUIRE(result(0, 1) == 32.0);
            REQUIRE(result(0, 2) == 50.0);
            REQUIRE(result(1, 0) == 32.0);
            REQUIRE(result(1, 1) == 77.0);
            REQUIRE(result(1, 2) == 122.0);
            REQUIRE(result(2, 0) == 50.0);
            REQUIRE(result(2, 1) == 122.0);
            REQUIRE(result(2, 2) == 194.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
        // true, false
        {
            gemm<true, false>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 66.0);
            REQUIRE(result(0, 1) == 78.0);
            REQUIRE(result(0, 2) == 90.0);
            REQUIRE(result(1, 0) == 78.0);
            REQUIRE(result(1, 1) == 93.0);
            REQUIRE(result(1, 2) == 108.0);
            REQUIRE(result(2, 0) == 90.0);
            REQUIRE(result(2, 1) == 108.0);
            REQUIRE(result(2, 2) == 126.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
        // true, true
        {
            gemm<true, true>(1.0, view, view, 0.0, &result);

            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 66.0);
            REQUIRE(result(0, 2) == 102.0);
            REQUIRE(result(1, 0) == 36.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 126.0);
            REQUIRE(result(2, 0) == 42.0);
            REQUIRE(result(2, 1) == 96.0);
            REQUIRE(result(2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched_d1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
    }
}

TEST_CASE("einsum1", "[tensor]") {
    using namespace einsums;
    using namespace TensorAlgebra;

    SECTION("ik=ij,jk") {
        Tensor<2> A{"A", 3, 3};
        Tensor<2> B{"B", 3, 3};
        Tensor<2> C{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
                B(i, j) = ij;
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{Index::i, Index::j}, &C, Indices{Index::i, Index::k}, A, Indices{Index::k, Index::j}, B));

        // println(A);
        // println(B);
        // println(C);

        /*[[ 30,  36,  42],
           [ 66,  81,  96],
           [102, 126, 150]]*/
        REQUIRE(C(0, 0) == 30.0);
        REQUIRE(C(0, 1) == 36.0);
        REQUIRE(C(0, 2) == 42.0);
        REQUIRE(C(1, 0) == 66.0);
        REQUIRE(C(1, 1) == 81.0);
        REQUIRE(C(1, 2) == 96.0);
        REQUIRE(C(2, 0) == 102.0);
        REQUIRE(C(2, 1) == 126.0);
        REQUIRE(C(2, 2) == 150.0);
    }

    SECTION("il=ijk,jkl") {
        Tensor<3> A{"A", 3, 3, 3};
        Tensor<3> B{"B", 3, 3, 3};
        Tensor<2> C{"C", 3, 3};

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
            einsum(Indices{Index::i, Index::l}, &C, Indices{Index::i, Index::j, Index::k}, A, Indices{Index::j, Index::k, Index::l}, B));

        // println(C);

        // array([[ 765.,  810.,  855.],
        //        [1818., 1944., 2070.],
        //        [2871., 3078., 3285.]])
        REQUIRE(C(0, 0) == 765.0);
        REQUIRE(C(0, 1) == 810.0);
        REQUIRE(C(0, 2) == 855.0);
        REQUIRE(C(1, 0) == 1818.0);
        REQUIRE(C(1, 1) == 1944.0);
        REQUIRE(C(1, 2) == 2070.0);
        REQUIRE(C(2, 0) == 2871.0);
        REQUIRE(C(2, 1) == 3078.0);
        REQUIRE(C(2, 2) == 3285.0);
    }
}

TEST_CASE("einsum TensorView", "[tensor]") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;

    SECTION("Subset View GEMM 7x3x3[4,:,:] -> [2,:,:]") {
        // Description: Obtain view [4,:,:] (3x3 view) perform GEMM and store result into
        // view [2,:,:] (3x3 view)
        const size_t d1_size = 7, d2_size = 3, d3_size = 3;
        const size_t d1 = 4;
        const size_t e1 = 2;
        const std::array<size_t, 6> untouched1{0, 1, 3, 4, 5, 6};

        Tensor<3> original = create_random_tensor("Original", d1_size, d2_size, d3_size);

        // Set submatrix to a set of known values
        for (size_t i = 0, ij = 1; i < 3; i++) {
            for (size_t j = 0; j < 3; j++, ij++) {
                original(d1, i, j) = static_cast<double>(ij);
            }
        }

        Tensor<3> copy = original;

        // Obtain a 3x3 view of original[4,:,:]
        TensorView<2> view = original(d1, All{}, All{});

        // Obtain a 3x3 view of original[2,:,:] to store the result
        TensorView<2> result = original(e1, All{}, All{});

        // false, false
        {
            // einsum("ik=ij,jk", &result, view, view);
            REQUIRE_NOTHROW(
                einsum(Indices{Index::i, Index::k}, &result, Indices{Index::i, Index::j}, view, Indices{Index::j, Index::k}, view));
            // gemm<false, false>(1.0, view, view, 0.0, &result);

            // Test against the view
            REQUIRE(result(0, 0) == 30.0);
            REQUIRE(result(0, 1) == 36.0);
            REQUIRE(result(0, 2) == 42.0);
            REQUIRE(result(1, 0) == 66.0);
            REQUIRE(result(1, 1) == 81.0);
            REQUIRE(result(1, 2) == 96.0);
            REQUIRE(result(2, 0) == 102.0);
            REQUIRE(result(2, 1) == 126.0);
            REQUIRE(result(2, 2) == 150.0);

            // Test the position in the original
            REQUIRE(original(2, 0, 0) == 30.0);
            REQUIRE(original(2, 0, 1) == 36.0);
            REQUIRE(original(2, 0, 2) == 42.0);
            REQUIRE(original(2, 1, 0) == 66.0);
            REQUIRE(original(2, 1, 1) == 81.0);
            REQUIRE(original(2, 1, 2) == 96.0);
            REQUIRE(original(2, 2, 0) == 102.0);
            REQUIRE(original(2, 2, 1) == 126.0);
            REQUIRE(original(2, 2, 2) == 150.0);

            // Test that the elements that shouldn't have been touched:
            for (size_t i : untouched1) {
                for (size_t j = 0; j < d2_size; j++) {
                    for (size_t k = 0; k < d3_size; k++) {
                        REQUIRE(original(i, j, k) == copy(i, j, k));
                    }
                }
            }
        }
    }
}

TEST_CASE("sort2") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace TensorAlgebra::Index;

    SECTION("Rank 2 - axpy") {
        Tensor<2> A{"A", 3, 3};
        Tensor<2> C{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        sort(Indices{i, j}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE(C(i, j) == A(i, j));
            }
        }

        TensorView<2> A_view{A, Dim<2>{2, 2}, Offset<2>{1, 1}};
        TensorView<2> C_view{C, Dim<2>{2, 2}, Offset<2>{1, 1}};

        sort(Indices{j, i}, &C_view, Indices{i, j}, A_view);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                if (i == 0 || j == 0)
                    REQUIRE(C(i, j) == A(i, j));
                else
                    REQUIRE(C(j, i) == A(i, j));
            }
        }
    }

    SECTION("Rank 2 - axpy (2)") {
        Tensor<2> A = create_random_tensor("A", 3, 3);
        Tensor<2> C0{"C", 3, 3};
        Tensor<2> C1{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = ij + A(i, j);
            }
        }

        sort(1.0, Indices{i, j}, &C0, 1.0, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE(C0(i, j) == C1(i, j));
            }
        }

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = 2.0 * ij + 0.5 * A(i, j);
            }
        }

        sort(2.0, Indices{i, j}, &C0, 0.5, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE(C0(i, j) == C1(i, j));
            }
        }
    }

    SECTION("Rank 2") {
        Tensor<2> A{"A", 3, 3};
        Tensor<2> C{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        sort(Indices{j, i}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE(C(j, i) == A(i, j));
            }
        }
    }

    SECTION("Rank 3") {
        Tensor<3> A{"A", 3, 3, 3};
        Tensor<3> B{"B", 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        sort(Indices{k, j, i}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE(B(k, j, i) == A(i, j, k));
                }
            }
        }

        sort(Indices{i, k, j}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE(B(i, k, j) == A(i, j, k));
                }
            }
        }

        sort(Indices{j, k, i}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE(B(j, k, i) == A(i, j, k));
                }
            }
        }

        sort(Indices{i, j, k}, &B, Indices{k, j, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE(B(i, j, k) == A(k, j, i));
                }
            }
        }
    }

    SECTION("Rank 4") {
        Tensor<4> A{"A", 3, 3, 3, 3};
        Tensor<4> B{"B", 3, 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++, ij++) {
                        A(i, j, k, l) = ij;
                    }
                }
            }
        }

        sort(0.0, Indices{i, l, k, j}, &B, 0.5, Indices{k, j, l, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        REQUIRE(B(i, l, k, j) == 0.5 * A(k, j, l, i));
                    }
                }
            }
        }
    }

    SECTION("Rank 5") {
        Tensor<5, float> A{"A", 3, 3, 3, 3, 3};
        Tensor<5, float> B{"B", 3, 3, 3, 3, 3};

        for (short i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        for (int m = 0; m < 3; m++, ij++) {
                            A(i, j, k, l, m) = ij;
                        }
                    }
                }
            }
        }

        sort(Indices{i, k, l, m, j}, &B, Indices{j, k, l, m, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        for (int m = 0; m < 3; m++) {
                            REQUIRE(B(i, k, l, m, j) == A(j, k, l, m, i));
                        }
                    }
                }
            }
        }
    }

    SECTION("Rank 2 - Different Sizes") {
        Tensor<2> A{"A", 3, 9};
        Tensor<2> B{"B", 9, 3};

        for (int i = 0, ij = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++, ij++) {
                A(i, j) = ij;
            }
        }

        sort(Indices{j, i}, &B, Indices{i, j}, A);
        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                REQUIRE(B(j, i) == A(i, j));
            }
        }
    }

    SECTION("Rank 3 - Different Sizes") {
        Tensor<3> A{"A", 2, 3, 4};
        Tensor<3> B{"B", 3, 4, 2};

        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        sort(Indices{j, k, i}, &B, Indices{i, j, k}, A);
        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    REQUIRE(B(j, k, i) == A(i, j, k));
                }
            }
        }
    }
}

TEST_CASE("einsum2") {
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    SECTION("3x3 <- 3x5 * 5x3") {
        Tensor<2> C0{"C0", 3, 3};
        Tensor<2> C1{"C1", 3, 3};
        Tensor<2> A = create_random_tensor("A", 3, 5);
        Tensor<2> B = create_random_tensor("B", 5, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, A, B, 0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3x3 <- 3x5 * 3x5") {
        Tensor<2> C0{"C0", 3, 3};
        Tensor<2> C1{"C1", 3, 3};
        Tensor<2> A = create_random_tensor("A", 3, 5);
        Tensor<2> B = create_random_tensor("B", 3, 5);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{j, k}, B));
        LinearAlgebra::gemm<false, true>(1.0, A, B, 0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3 <- 3x5 * 5") {
        Tensor<1> C0{"C0", 3};
        Tensor<1> C1{"C1", 3};
        Tensor<2> A = create_random_tensor("A", 3, 5);
        Tensor<1> B = create_random_tensor("B", 5);

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j}, A, Indices{j}, B));
        LinearAlgebra::gemv<false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            REQUIRE_THAT(C0(i0), Catch::Matchers::WithinAbs(C1(i0), 0.001));
        }
    }

    SECTION("3 <- 3x4x5 * 4x3x5") {
        Tensor<1> C0{"C0", 3};
        Tensor<1> C1{"C1", 3};
        Tensor<3> A = create_random_tensor("A", 3, 4, 5);
        Tensor<3> B = create_random_tensor("B", 4, 3, 5);

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));

        for (size_t i0 = 0; i0 < 3; i0++) {
            double sum{0};

            for (size_t j0 = 0; j0 < 4; j0++) {
                for (size_t k0 = 0; k0 < 5; k0++) {
                    sum += A(i0, j0, k0) * B(j0, i0, k0);
                }
            }
            C1(i0) = sum;
        }

        for (size_t i0 = 0; i0 < 3; i0++) {
            REQUIRE_THAT(C0(i0), Catch::Matchers::WithinRel(C1(i0), 0.0001));
        }
    }

    SECTION("3x5 <- 3x4x5 * 4x3x5") {
        Tensor<2> C0{"C0", 3, 5};
        Tensor<2> C1{"C1", 3, 5};
        Tensor<3> A = create_random_tensor("A", 3, 4, 5);
        Tensor<3> B = create_random_tensor("B", 4, 3, 5);

        // Timer::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, k}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // Timer::pop();

        // Timer::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                double sum{0};
                for (size_t j0 = 0; j0 < 4; j0++) {

                    sum += A(i0, j0, k0) * B(j0, i0, k0);
                }
                C1(i0, k0) = sum;
            }
        }
        // Timer::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3, l <- 3x4x5 * 4x3x5") {
        Tensor<2> C0{"C0", 3, 5};
        Tensor<2> C1{"C1", 3, 5};
        Tensor<3> A = create_random_tensor("A", 3, 4, 5);
        Tensor<3> B = create_random_tensor("B", 4, 3, 5);

        // Timer::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // Timer::pop();

        // Timer::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                for (size_t l0 = 0; l0 < 5; l0++) {
                    double sum{0};
                    for (size_t j0 = 0; j0 < 4; j0++) {

                        sum += A(i0, j0, k0) * B(j0, i0, k0);
                    }
                    C1(i0, l0) += sum;
                }
            }
        }
        // Timer::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                // REQUIRE(C0(i0, j0) == C1(i0, j0));?
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0000001));
            }
        }
    }

    // Timer::report();
    // Timer::finalize();
}

TEST_CASE("einsum3") {
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    // Timer::initialize();

    SECTION("3x3 <- 3x5 * 5x3") {
        Tensor<2> C0{"C0", 3, 3};
        Tensor<2> C1{"C1", 3, 3};
        Tensor<2> A = create_random_tensor("A", 3, 5);
        Tensor<2> B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, A, B, 0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3x3x3x3 <- 3x3x3x3 * 3x3") {
        // This one is to represent a two-electron integral transformation
        Tensor<4> gMO0{"g0", 3, 3, 3, 3};
        Tensor<4> gMO1{"g1", 3, 3, 3, 3};
        Tensor<4> A = create_random_tensor("A", 3, 3, 3, 3);
        Tensor<2> B = create_random_tensor("B", 3, 3);

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
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
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
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                auto vgMO0 = gMO0(i0, j0, All{}, All{});
                REQUIRE_NOTHROW(einsum(Indices{k, l}, &vgMO0, Indices{p, l}, A(i0, j0, All{}, All{}), Indices{p, k}, B));
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }
    }

    // Timer::report();
    // Timer::finalize();
}

TEST_CASE("einsum4") {
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    // Timer::initialize();
    SECTION("3x3x3x3 <- 3x3x3x3 * 3x3") {
        // This one is to represent a two-electron integral transformation
        Tensor<4> gMO0{"g0", 3, 3, 3, 3};
        Tensor<4> gMO1{"g1", 3, 3, 3, 3};
        Tensor<4> A = create_random_tensor("A", 3, 3, 3, 3);
        Tensor<2> B = create_random_tensor("B", 3, 3);

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
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
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
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

#if 0
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                auto vgMO0 = gMO0(i0, j0, All{}, All{});
                TensorAlgebra::einsum(Indices{k, s}, &vgMO0, Indices{r, s}, A(i0, j0, All{}, All{}), Indices{r, k}, B);
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        // println("i0 %lu j0 %lu k0 %lu l0 %lu, gMO0 %lf, gMO1 %lf", i0, j0, k0, l0, gMO0(i0, j0, k0, l0),
                        // gMO1(i0, j0, k0, l0));
                        REQUIRE(gMO0(i0, j0, k0, l0) == gMO1(i0, j0, k0, l0));
                    }
                }
            }
        }
#endif
    }

    // Timer::report();
    // Timer::finalize();
}

TEST_CASE("IntegralTransformation") {
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    // Timer::initialize();
    // SECTION("3x3x3x3 <- 3x3x3x3 * 3x3 * 3x3 * 3x3 * 3x3") {
    //     Tensor<2> C1 = create_random_tensor("C1", 4, 2);
    //     Tensor<2> C2 = create_random_tensor("C2", 4, 2);
    //     Tensor<2> C3 = create_random_tensor("C3", 4, 3);
    //     Tensor<2> C4 = create_random_tensor("C4", 4, 4);

    //     Tensor<4> true_answer{"true", 2, 2, 3, 4};
    //     Tensor<4> memory_ao = create_random_tensor("ao", 4, 4, 4, 4);
    //     DiskTensor<4> disk_ao{State::data, "ao", 4, 4, 4, 4};

    //     // #pragma omp parallel for collapse(8)
    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {

    //                     for (size_t p0 = 0; p0 < C1.dim(0); p0++) {
    //                         for (size_t q0 = 0; q0 < C2.dim(0); q0++) {
    //                             for (size_t r0 = 0; r0 < C3.dim(0); r0++) {
    //                                 for (size_t s0 = 0; s0 < C4.dim(0); s0++) {
    //                                     true_answer(i0, j0, k0, l0) +=
    //                                         C1(p0, i0) * C2(q0, j0) * C3(r0, k0) * C4(s0, l0) * memory_ao(p0, q0, r0, s0);
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // Save our inital memory_ao to disk
    //     write(State::data, C1);
    //     write(State::data, C2);
    //     write(State::data, C3);
    //     write(State::data, C4);

    //     disk_ao(All{}, All{}, All{}, All{}) = memory_ao;
    //     auto temp = disk_ao(All{}, All{}, All{}, All{});
    //     auto &temp2 = temp.get();

    //     // Ensure the data was saved to disk correctly
    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {
    //                     REQUIRE(temp2(i0, j0, k0, l0) == memory_ao(i0, j0, k0, l0));
    //                 }
    //             }
    //         }
    //     }

    //     auto memory_result = transformation("mo", memory_ao, C1, C2, C3, C4);
    //     auto disk_result = transformation("mo", disk_ao, C1, C2, C3, C4);

    //     // Make sure the memory and disk results match
    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {
    //                     REQUIRE_THAT(memory_result(i0, j0, k0, l0), Catch::Matchers::WithinRel(true_answer(i0, j0, k0, l0), 0.0000001));
    //                 }
    //             }
    //         }
    //     }

    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             auto disk_view = disk_result(i0, j0, All{}, All{});
    //             auto &disk_tensor = disk_view.get();
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {
    //                     REQUIRE_THAT(disk_tensor(k0, l0), Catch::Matchers::WithinRel(true_answer(i0, j0, k0, l0), 0.0000001));
    //                 }
    //             }
    //         }
    //     }
    // }

    SECTION("R2 <- R3 * R3") {
        Tensor<2> W_mi = create_random_tensor("W_mi", 4, 4);
        Tensor<3> g_m = create_random_tensor("g_m", 4, 8, 8);
        Tensor<3> t_i = create_random_tensor("t_i", 4, 8, 8);

        // println(W_mi);
        // println(g_m);
        // println(t_i);

        REQUIRE_NOTHROW(einsum(1.0, Indices{Index::n, Index::j}, &W_mi, 0.25, Indices{Index::n, Index::e, Index::f}, g_m,
                               Indices{Index::j, Index::e, Index::f}, t_i));
    }
}

TEST_CASE("Hadamard") {
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- i,i * j*j") {
        Tensor<2> A = create_random_tensor("A", _i, _i);
        Tensor<2> B = create_random_tensor("B", _j, _j);
        Tensor<2> C{"C", _i, _j};
        Tensor<2> C0{"C0", _i, _j};
        C0.zero();

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
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j <- i,i,j * j,j,i") {
        Tensor<3> A = create_random_tensor("A", _i, _i, _j);
        Tensor<3> B = create_random_tensor("B", _j, _j, _i);
        Tensor<2> C{"C", _i, _j};
        Tensor<2> C0{"C0", _i, _j};
        C0.zero();

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
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j <- i,j,i * j,i,j") {
        Tensor<3> A = create_random_tensor("A", _i, _j, _i);
        Tensor<3> B = create_random_tensor("B", _j, _i, _j);
        Tensor<2> C{"C", _i, _j};
        Tensor<2> C0{"C0", _i, _j};
        C0.zero();

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
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j,i <- i,j,i * j,i,j") {
        Tensor<3> A = create_random_tensor("A", _i, _j, _i);
        Tensor<3> B = create_random_tensor("B", _j, _i, _j);
        Tensor<3> C{"C", _i, _j, _i};
        Tensor<3> C0{"C0", _i, _j, _i};
        C0.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CHECK_THAT(C(i0, j0, i0), Catch::Matchers::WithinRel(C0(i0, j0, i0), 0.00001));
            }
        }
    }

    SECTION("i,i,i <- i,j,i * j,i,j") {
        Tensor<3> A = create_random_tensor("A", _i, _j, _i);
        Tensor<3> B = create_random_tensor("B", _j, _i, _j);
        Tensor<3> C{"C", _i, _i, _i};
        Tensor<3> C0{"C0", _i, _i, _i};
        C0.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, i0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, i, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            CHECK_THAT(C(i0, i0, i0), Catch::Matchers::WithinRel(C0(i0, i0, i0), 0.00001));
        }
    }

    SECTION("i,i <- i,j,k * j,i,k") {
        Tensor<3> A = create_random_tensor("A", _i, _j, _k);
        Tensor<3> B = create_random_tensor("B", _j, _i, _k);
        Tensor<2> C{"C", _i, _i};
        Tensor<2> C0{"C0", _i, _i};
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
                CHECK_THAT(C(i0, j0), Catch::Matchers::WithinRel(C0(i0, j0), 0.00001));
            }
        }
    }
}

TEST_CASE("unique_ptr") {
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    SECTION("C") {
        auto C0 = std::make_unique<Tensor<2>>("C0", 3, 3);
        Tensor<2> C1{"C1", 3, 3};
        Tensor<2> A = create_random_tensor("A", 3, 5);
        Tensor<2> B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, A, B, 0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("A") {
        Tensor<2> C0{"C0", 3, 3};
        Tensor<2> C1{"C1", 3, 3};
        auto A = std::make_unique<Tensor<2>>(create_random_tensor("A", 3, 5));
        Tensor<2> B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, *A, B, 0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("B") {
        Tensor<2> C0{"C0", 3, 3};
        Tensor<2> C1{"C1", 3, 3};
        Tensor<2> A = create_random_tensor("A", 5, 3);
        auto B = std::make_unique<Tensor<2>>(create_random_tensor("B", 3, 5));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, A, *B, 0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("AB") {
        Tensor<2> C0{"C0", 3, 3};
        Tensor<2> C1{"C1", 3, 3};
        auto A = std::make_unique<Tensor<2>>(create_random_tensor("A", 3, 5));
        auto B = std::make_unique<Tensor<2>>(create_random_tensor("B", 3, 5));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, *A, *B, 0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CA") {
        auto C0 = std::make_unique<Tensor<2>>("C0", 3, 3);
        Tensor<2> C1{"C1", 3, 3};
        auto A = std::make_unique<Tensor<2>>(create_random_tensor("A", 3, 5));
        Tensor<2> B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, *A, B, 0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CB") {
        auto C0 = std::make_unique<Tensor<2>>("C0", 3, 3);
        Tensor<2> C1{"C1", 3, 3};
        Tensor<2> A = create_random_tensor("A", 5, 3);
        auto B = std::make_unique<Tensor<2>>(create_random_tensor("B", 3, 5));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, A, *B, 0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CAB") {
        auto C0 = std::make_unique<Tensor<2>>("C0", 3, 3);
        Tensor<2> C1{"C1", 3, 3};
        auto A = std::make_unique<Tensor<2>>(create_random_tensor("A", 3, 5));
        auto B = std::make_unique<Tensor<2>>(create_random_tensor("B", 3, 5));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        LinearAlgebra::gemm<false, false>(1.0, *A, *B, 0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }
}

TEST_CASE("Transpose C", "[einsum]") {
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- j,k * k,i === true, false, false") {
        Tensor<2> A = create_random_tensor("A", _j, _k);
        Tensor<2> B = create_random_tensor("B", _k, _i);
        Tensor<2> C{"C", _i, _j};
        Tensor<2> C0{"C0", _i, _j};
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
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * k,i === true, true, false") {
        Tensor<2> A = create_random_tensor("A", _k, _j);
        Tensor<2> B = create_random_tensor("B", _k, _i);
        Tensor<2> C{"C", _i, _j};
        Tensor<2> C0{"C0", _i, _j};
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
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- j,k * i,k === true, false, true") {
        Tensor<2> A = create_random_tensor("A", _j, _k);
        Tensor<2> B = create_random_tensor("B", _i, _k);
        Tensor<2> C{"C", _i, _j};
        Tensor<2> C0{"C0", _i, _j};
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
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * i,k === true, true, true") {
        Tensor<2> A = create_random_tensor("A", _k, _j);
        Tensor<2> B = create_random_tensor("B", _i, _k);
        Tensor<2> C{"C", _i, _j};
        Tensor<2> C0{"C0", _i, _j};
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
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("Wmnij <- 0.25 t_ijef * g_mnef") {
        size_t _m = 12, _n = 12, _i = 5, _j = 5, _e = 7, _f = 7;

        Tensor<4> Wmnij{"Wmnij", _m, _n, _i, _j};
        Tensor<4> W0{"Wmnij", _m, _n, _i, _j};

        Tensor<4> t_oovv = create_random_tensor("t_oovv", _i, _j, _e, _f);
        Tensor<4> g_oovv = create_random_tensor("g_oovv", _m, _n, _e, _f);

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
                        REQUIRE_THAT(Wmnij(m0, n0, i0, j0), Catch::Matchers::WithinAbs(W0(m0, n0, i0, j0), 0.001));
                    }
                }
            }
        }
    }
}

TEST_CASE("gemv") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace einsums::TensorAlgebra::Index;

    SECTION("check") {
        size_t _p = 7, _q = 7, _r = 7, _s = 7;

        Tensor<4> g = create_random_tensor("g", _p, _q, _r, _s);
        Tensor<2> D = create_random_tensor("d", _r, _s);

        Tensor<2> F{"F", _p, _q};
        Tensor<2> F0{"F0", _p, _q};

        REQUIRE_NOTHROW(einsum(1.0, Indices{p, q}, &F0, 2.0, Indices{p, q, r, s}, g, Indices{r, s}, D));

        TensorView<2> gv{g, Dim<2>{_p * _q, _r * _s}};
        TensorView<1> dv{D, Dim<1>{_r * _s}};
        TensorView<1> Fv{F, Dim<1>{_p * _q}};

        LinearAlgebra::gemv<false>(2.0, gv, dv, 1.0, &Fv);

        // println(F0);
        // println(F);
    }
}

TEST_CASE("TensorView einsum") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace einsums::TensorAlgebra::Index;

    // Test if everything passed to einsum is a TensorView.
    Tensor<2> A = create_random_tensor("A", 3, 5);
    Tensor<2> B = create_random_tensor("B", 3, 5);
    TensorView<2> A_view{A, Dim<2>{3, 3}};
    TensorView<2> B_view{B, Dim<2>{3, 3}, Offset<2>{0, 2}};

    Tensor<2> C{"C2", 10, 10};
    TensorView<2> C_view{C, Dim<2>{3, 3}, Offset<2>{5, 5}};

    // To perform the test we make an explicit copy of the TensorViews into their own Tensors
    Tensor<2> A_copy{"A copy", 3, 3};
    Tensor<2> B_copy{"B copy", 3, 3};

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            A_copy(x, y) = A_view(x, y);
            B_copy(x, y) = B_view(x, y);
        }
    }

    // The target solution is determined from not using views
    Tensor<2> C_solution{"C solution", 3, 3};
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_solution, Indices{i, k}, A_copy, Indices{j, k}, B_copy));

    // einsum where everything is a TensorView
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_view, Indices{i, k}, A_view, Indices{j, k}, B_view));

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            REQUIRE_THAT(C_view(x, y), Catch::Matchers::WithinAbs(C_solution(x, y), 0.001));
            REQUIRE_THAT(C(x + 5, y + 5), Catch::Matchers::WithinAbs(C_solution(x, y), 0.001));
        }
    }
}

TEST_CASE("outer product") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace einsums::TensorAlgebra::Index;

    SECTION("1 * 1 -> 2") {
        Tensor<1> A = create_random_tensor("A", 3);
        Tensor<1> B = create_random_tensor("B", 3);
        Tensor<2> C{"C", 3, 3};
        C.set_all(0.0);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }
    }

    SECTION("2 * 1 -> 3") {
        Tensor<2> A = create_random_tensor("A", 3, 3);
        Tensor<1> B = create_random_tensor("B", 3);
        Tensor<3> C{"C", 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(x, y, z), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{k}, B, Indices{i, j}, A));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        Tensor<2> A = create_random_tensor("A", 3, 3);
        Tensor<2> B = create_random_tensor("B", 3, 3);
        Tensor<4> C{"C", 3, 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, A, Indices{k, l}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(w, x) * B(y, z), 0.001));
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
                        REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(y, z) * B(w, x), 0.001));
                    }
                }
            }
        }
    }
}

TEST_CASE("view outer product") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace einsums::TensorAlgebra::Index;

    SECTION("1 * 1 -> 2") {
        Tensor<1> A = create_random_tensor("A", 6);
        Tensor<1> B = create_random_tensor("B", 6);

        auto vA = TensorView<1>(A, Dim<1>{3}, Offset<1>{3});
        auto vB = TensorView<1>(B, Dim<1>{3});
        Tensor<2> C{"C", 3, 3};
        C.set_all(0.0);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, vA, Indices{j}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(vA(x) * vB(y), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, vA, Indices{i}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(vA(y) * vB(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, vA, Indices{i}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(vA(y) * vB(x), 0.001));
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, vA, Indices{j}, vB));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(vA(x) * vB(y), 0.001));
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        Tensor<2> A = create_random_tensor("A", 9, 9);
        Tensor<2> B = create_random_tensor("B", 12, 12);
        auto vA = TensorView{A, Dim<2>{3, 3}, Offset<2>{6, 3}};
        auto vB = TensorView{B, Dim<2>{3, 3}, Offset<2>{5, 7}};
        Tensor<4> C{"C", 3, 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, vA, Indices{k, l}, vB));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(vA(w, x) * vB(y, z), 0.001));
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
                        REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(vA(y, z) * vB(w, x), 0.001));
                    }
                }
            }
        }
    }
}

TEST_CASE("element transform") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace einsums::TensorAlgebra::Index;

    SECTION("tensor") {
        Tensor<4> A = create_random_tensor("A", 32, 32, 32, 32);
        Tensor<4> Acopy = A;

        element_transform(&A, [](double val) -> double { return 1.0 / val; });

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(1.0 / Acopy(w, x, y, z), 0.001));
                    }
                }
            }
        }
    }

    SECTION("smartptr tensor") {
        auto A = std::make_unique<Tensor<4>>("A", 32, 32, 32, 32);

        element_transform(&A, [](double val) -> double { return 1.0 / val; });
    }
}

TEST_CASE("element") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace einsums::TensorAlgebra::Index;

    SECTION("1") {
        Tensor<4> A = create_random_tensor("A", 10, 10, 10, 10);
        Tensor<4> Acopy = A;

        Tensor<4> B = create_random_tensor("B", 10, 10, 10, 10);

        element([](double const &Aval, double const &Bval) -> double { return Aval + Bval; }, &A, B);

        for (int w = 0; w < 10; w++) {
            for (int x = 0; x < 10; x++) {
                for (int y = 0; y < 10; y++) {
                    for (int z = 0; z < 10; z++) {
                        REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(Acopy(w, x, y, z) + B(w, x, y, z), 0.001));
                    }
                }
            }
        }
    }

    SECTION("2") {
        Tensor<4> A = create_random_tensor("A", 10, 10, 10, 10);
        Tensor<4> Acopy = A;

        Tensor<4> B = create_random_tensor("B", 10, 10, 10, 10);
        Tensor<4> C = create_random_tensor("C", 10, 10, 10, 10);

        element([](double const &Aval, double const &Bval, double const &Cval) -> double { return Aval + Bval + Cval; }, &A, B, C);

        for (int w = 0; w < 10; w++) {
            for (int x = 0; x < 10; x++) {
                for (int y = 0; y < 10; y++) {
                    for (int z = 0; z < 10; z++) {
                        REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(Acopy(w, x, y, z) + B(w, x, y, z) + C(w, x, y, z), 0.001));
                    }
                }
            }
        }
    }

    // SECTION("2 - error") {

    //     Tensor<4> A = create_random_tensor("A", 10, 10, 10, 10);
    //     Tensor<4> Acopy = A;

    //     Tensor<4> B = create_random_tensor("B", 10, 10, 10, 9);
    //     Tensor<4> C = create_random_tensor("C", 10, 10, 10, 10);

    //     element(
    //         &A, [](double const &Aval, double const &Bval, double const &Cval) -> double { return Aval + Bval + Cval; }, B, C);
    // }
}

TEST_CASE("F12 - V term") {
    using namespace einsums;
    using namespace einsums::TensorAlgebra;
    using namespace einsums::TensorAlgebra::Index;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{1}, ncabs{4}, nobs{2};
    int nall{nobs + ncabs};

    auto F = create_incremented_tensor("F", nall, nall, nall, nall);
    auto G = create_incremented_tensor("G", nall, nall, nall, nall);

    TensorView<4> F_ooco{F, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    TensorView<4> F_oooc{F, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    TensorView<4> F_oopq{F, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};
    TensorView<4> G_ooco{G, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    TensorView<4> G_oooc{G, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    TensorView<4> G_oopq{G, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};

    Tensor ijkl_1 = Tensor{"Einsum Temp 1", nocc, nocc, nocc, nocc};
    Tensor ijkl_2 = Tensor{"Einsum Temp 2", nocc, nocc, nocc, nocc};
    Tensor ijkl_3 = Tensor{"Einsum Temp 3", nocc, nocc, nocc, nocc};

    ijkl_1.set_all(0.0);
    ijkl_2.set_all(0.0);
    ijkl_3.set_all(0.0);

    Tensor result = Tensor{"Result", nocc, nocc, nocc, nocc};
    Tensor result2 = Tensor{"Result2", nocc, nocc, nocc, nocc};

    println(F);
    println(G);

    omp_set_num_threads(1);
    einsum(Indices{i, j, k, l}, &ijkl_1, Indices{i, j, p, n}, G_ooco, Indices{k, l, p, n}, F_ooco);
    // einsum(Indices{i, j, k, l}, &ijkl_2, Indices{i, j, m, q}, G_oooc, Indices{k, l, m, q}, F_oooc);
    // einsum(Indices{i, j, k, l}, &ijkl_3, Indices{i, j, p, q}, G_oopq, Indices{k, l, p, q}, F_oopq);

    result.set_all(0.0);
    result2.set_all(0.0);
    Timer::push("raw for loops");
    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    for (size_t _p = 0; _p < ncabs; _p++) {
                        for (size_t _n = 0; _n < nocc; _n++) {
                            println("A({}, {}, {}, {}) = {}", _i, _j, _p, _n, G_ooco(_i, _j, _p, _n));
                            println("B({}, {}, {}, {}) = {}", _k, _l, _p, _n, F_ooco(_k, _l, _p, _n));

                            result(_i, _j, _k, _l) += G(_i, _j, nobs + _p, _n) * F(_k, _l, nobs + _p, _n);
                            result2(_i, _j, _k, _l) += G_ooco(_i, _j, _p, _n) * F_ooco(_k, _l, _p, _n);
                        }
                    }
                }
            }
        }
    }
    Timer::pop();

    println(result);
    println(ijkl_1);

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(result2(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(ijkl_1(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }
}