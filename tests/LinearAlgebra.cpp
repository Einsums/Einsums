#include "einsums/LinearAlgebra.hpp"

#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/Utilities.hpp"

#include <catch2/catch_all.hpp>
#include <filesystem>

using namespace einsums;
namespace {
template <typename T>
auto reconstruct(const Tensor<T, 2> &u, const Tensor<T, 1> &s, const Tensor<T, 2> &vt, size_t N, size_t LDA) -> Tensor<T, 2> {
    using namespace einsums::tensor_algebra;

    auto half = create_tensor<T>("sigma", N, LDA);
    zero(half);
    auto new_a = create_tensor<T>("new a", N, LDA);
    zero(new_a);
    auto sigma = diagonal_like(s, new_a);

    einsum(Indices{index::i, index::j}, &half, Indices{index::i, index::k}, u, Indices{index::k, index::j}, sigma);
    einsum(Indices{index::i, index::j}, &new_a, Indices{index::i, index::k}, half, Indices{index::k, index::j}, vt);

    return new_a;
}
} // namespace

template <typename T>
void gesvd_test() {
    /*
       DGESVD Example.
       ==============

       Program computes the singular value decomposition of a general
       rectangular matrix A:

         8.79   9.93   9.83   5.45   3.16
         6.11   6.91   5.04  -0.27   7.98
        -9.15  -7.93   4.86   4.85   3.01
         9.57   1.64   8.83   0.74   5.80
        -3.49   4.02   9.80  10.00   4.27
         9.84   0.15  -8.99  -6.02  -5.31

       Description.
       ============

       The routine computes the singular value decomposition (SVD) of a real
       m-by-n matrix A, optionally computing the left and/or right singular
       vectors. The SVD is written as

       A = U*SIGMA*VT

       where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
       diagonal elements, U is an m-by-m orthogonal matrix and VT (V transposed)
       is an n-by-n orthogonal matrix. The diagonal elements of SIGMA
       are the singular values of A; they are real and non-negative, and are
       returned in descending order. The first min(m, n) columns of U and V are
       the left and right singular vectors of A.

       Note that the routine returns VT, not V.

       Example Program Results.
       ========================

     DGESVD Example Program Results

     Singular values
      27.47  22.64   8.56   5.99   2.01

     Left singular vectors (stored columnwise)
      -0.59   0.26   0.36   0.31   0.23
      -0.40   0.24  -0.22  -0.75  -0.36
      -0.03  -0.60  -0.45   0.23  -0.31
      -0.43   0.24  -0.69   0.33   0.16
      -0.47  -0.35   0.39   0.16  -0.52
       0.29   0.58  -0.02   0.38  -0.65

     Right singular vectors (stored rowwise)
      -0.25  -0.40  -0.69  -0.37  -0.41
       0.81   0.36  -0.25  -0.37  -0.10
      -0.26   0.70  -0.22   0.39  -0.49
       0.40  -0.45   0.25   0.43  -0.62
      -0.22   0.14   0.59  -0.63  -0.44
    */
    constexpr int M{6};
    constexpr int N{5};
    constexpr int LDA{M};
    constexpr int LDU{M};
    constexpr int LDVT{N};

    using namespace einsums;

    auto a = create_tensor<T>("a", N, LDA);

    a.vector_data() = std::vector<T, einsums::AlignedAllocator<T, 64>>{8.79, 6.11, -9.15, 9.57,  -3.49, 9.84, 9.93, 6.91,  -7.93, 1.64,
                                                                       4.02, 0.15, 9.83,  5.04,  4.86,  8.83, 9.80, -8.99, 5.45,  -0.27,
                                                                       4.85, 0.74, 10.00, -6.02, 3.16,  7.98, 3.01, 5.80,  4.27,  -5.31};

    auto [u, s, vt] = linear_algebra::svd(a);

    // Using u, s, and vt reconstruct a and test a against the reconstructed a
    auto new_a = reconstruct(u, s, vt, N, LDA);

    CHECK_THAT(new_a.vector_data(), Catch::Matchers::Approx(a.vector_data()).margin(0.0001));
}

TEST_CASE("gesvd") {
    SECTION("float") {
        gesvd_test<float>();
    }
    SECTION("double") {
        gesvd_test<double>();
    }
}

template <typename T>
void gesdd_test() {
    /*
   DGESDD Example.
   ==============

   Program computes the singular value decomposition of a general
   rectangular matrix A using a divide and conquer method, where A is:

     7.52  -1.10  -7.95   1.08
    -0.76   0.62   9.34  -7.10
     5.13   6.62  -5.66   0.87
    -4.75   8.52   5.75   5.30
     1.33   4.91  -5.49  -3.52
    -2.40  -6.77   2.34   3.95

   Description.
   ============

   The routine computes the singular value decomposition (SVD) of a real
   m-by-n matrix A, optionally computing the left and/or right singular
   vectors. If singular vectors are desired, it uses a divide and conquer
   algorithm. The SVD is written as

   A = U*SIGMA*VT

   where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
   diagonal elements, U is an m-by-m orthogonal matrix and VT (V transposed)
   is an n-by-n orthogonal matrix. The diagonal elements of SIGMA
   are the singular values of A; they are real and non-negative, and are
   returned in descending order. The first min(m, n) columns of U and V are
   the left and right singular vectors of A.

   Note that the routine returns VT, not V.

   Example Program Results.
   ========================

 DGESDD Example Program Results

 Singular values
  18.37  13.63  10.85   4.49

 Left singular vectors (stored columnwise)
  -0.57   0.18   0.01   0.53
   0.46  -0.11  -0.72   0.42
  -0.45  -0.41   0.00   0.36
   0.33  -0.69   0.49   0.19
  -0.32  -0.31  -0.28  -0.61
   0.21   0.46   0.39   0.09

 Right singular vectors (stored rowwise)
  -0.52  -0.12   0.85  -0.03
   0.08  -0.99  -0.09  -0.01
  -0.28  -0.02  -0.14   0.95
   0.81   0.01   0.50   0.31
*/
    constexpr int M{6};
    constexpr int N{4};
    constexpr int LDA{M};
    constexpr int LDU{M};
    constexpr int LDVT{N};

    using namespace einsums;

    auto a = create_tensor<T>("a", N, LDA);

    a.vector_data() =
        std::vector<T, einsums::AlignedAllocator<T, 64>>{7.52,  -0.76, 5.13,  -4.75, 1.33,  -2.40, -1.10, 0.62,  6.62, 8.52, 4.91,  -6.77,
                                                         -7.95, 9.34,  -5.66, 5.75,  -5.49, 2.34,  1.08,  -7.10, 0.87, 5.30, -3.52, 3.95};

    auto [u, s, vt] = linear_algebra::svd_dd(a);

    // Using u, s, and vt reconstruct a and test a against the reconstructed a
    auto new_a = reconstruct(u, s, vt, N, LDA);

    CHECK_THAT(new_a.vector_data(), Catch::Matchers::Approx(a.vector_data()).margin(0.0001));
}

TEST_CASE("gesdd") {
    SECTION("double") {
        gesdd_test<double>();
    }
    SECTION("float") {
        gesdd_test<float>();
    }
}

template <typename T>
void gesv_test() {
    /*
       LAPACKE_dgesv Example.
       ======================

       The program computes the solution to the system of linear
       equations with a square matrix A and multiple
       right-hand sides B, where A is the coefficient matrix:

         6.80  -6.05  -0.45   8.32  -9.67
        -2.11  -3.30   2.58   2.71  -5.14
         5.66   5.36  -2.70   4.35  -7.26
         5.97  -4.44   0.27  -7.17   6.08
         8.23   1.08   9.04   2.14  -6.87

       and B is the right-hand side matrix:

         4.02  -1.56   9.81
         6.19   4.00  -4.09
        -8.22  -8.67  -4.57
        -7.57   1.75  -8.61
        -3.03   2.86   8.99

       Description.
       ============

       The routine solves for X the system of linear equations A*X = B,
       where A is an n-by-n matrix, the columns of matrix B are individual
       right-hand sides, and the columns of X are the corresponding
       solutions.

       The LU decomposition with partial pivoting and row interchanges is
       used to factor A as A = P*L*U, where P is a permutation matrix, L
       is unit lower triangular, and U is upper triangular. The factored
       form of A is then used to solve the system of equations A*X = B.

       Example Program Results.
       ========================

     LAPACKE_dgesv (column-major, high-level) Example Program Results

     Solution
      -0.80  -0.39   0.96
      -0.70  -0.55   0.22
       0.59   0.84   1.90
       1.32  -0.10   5.36
       0.57   0.11   4.04

     Details of LU factorization
       8.23   1.08   9.04   2.14  -6.87
       0.83  -6.94  -7.92   6.55  -3.99
       0.69  -0.67 -14.18   7.24  -5.19
       0.73   0.75   0.02 -13.82  14.19
      -0.26   0.44  -0.59  -0.34  -3.43

     Pivot indices
          5      5      3      4      5
        */

    constexpr int N{5};
    constexpr int NRHS{3};
    constexpr int LDA{N};
    constexpr int LDB{N};

    using namespace einsums;

    auto a = create_tensor<T>("a", N, LDA);
    auto b = create_tensor<T>("b", NRHS, LDB);

    a.vector_data() = std::vector<T, einsums::AlignedAllocator<T, 64>>{6.80,  -2.11, 5.66,  5.97,  8.23,  -6.05, -3.30, 5.36, -4.44,
                                                                       1.08,  -0.45, 2.58,  -2.70, 0.27,  9.04,  8.32,  2.71, 4.35,
                                                                       -7.17, 2.14,  -9.67, -5.14, -7.26, 6.08,  -6.87};
    b.vector_data() = std::vector<T, einsums::AlignedAllocator<T, 64>>{4.02, 6.19, -8.22, -7.57, -3.03, -1.56, 4.00, -8.67,
                                                                       1.75, 2.86, 9.81,  -4.09, -4.57, -8.61, 8.99};

    linear_algebra::gesv(&a, &b);

    CHECK_THAT(a.vector_data(),
               Catch::Matchers::Approx(std::vector<T, einsums::AlignedAllocator<T, 64>>{
                                           8.23000000,  0.82624544,  0.68772783,  0.72539490, -0.25637910,  1.08000000,   -6.94234508,
                                           -0.66508563, 0.75240087,  0.43545957,  9.04000000, -7.91925881,  -14.18404477, 0.02320302,
                                           -0.58842060, 2.14000000,  6.55183475,  7.23579360, -13.81984350, -0.33743379,  -6.87000000,
                                           -3.99369380, -5.19145820, 14.18877913, -3.42921969})
                   .margin(0.00001));
    CHECK_THAT(b.vector_data(),
               Catch::Matchers::Approx(std::vector<T, einsums::AlignedAllocator<T, 64>>{
                                           -0.80071403, -0.69524338, 0.59391499, 1.32172561, 0.56575620, -0.38962139, -0.55442713,
                                           0.84222739, -0.10380185, 0.10571095, 0.95546491, 0.22065963, 1.90063673, 5.35766149, 4.04060266})
                   .margin(0.00001));
}

TEST_CASE("gesv") {
    SECTION("float") {
        gesv_test<float>();
    }
    SECTION("double") {
        gesv_test<double>();
    }
    SECTION("complex<float>") {
        // gesv_test<std::complex<float>>();
    }
}

template <typename T>
void gemm_test_1() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A  = create_incremented_tensor<T>("a", 3, 3);
    auto B  = create_incremented_tensor<T>("b", 3, 3);
    auto C  = create_tensor<T>("c", 3, 3);
    auto C0 = create_tensor<T>("C0", 3, 3);
    zero(C);
    zero(C0);

    // Perform basic matrix multiplication
    einsums::linear_algebra::gemm<false, false>(T{1.0}, A, B, T{0.0}, &C);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 3; k++) {
                C0(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(C0(i, j) == C(i, j));
        }
    }
}

TEST_CASE("gemm_1") {
    SECTION("float") {
        gemm_test_1<float>();
    }

    SECTION("double") {
        gemm_test_1<double>();
    }

    SECTION("complex float") {
        gemm_test_1<std::complex<float>>();
    }

    SECTION("complex double") {
        gemm_test_1<std::complex<double>>();
    }
}

template <typename T>
void gemm_test_2() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A  = create_incremented_tensor<T>("a", 3, 2);
    auto B  = create_incremented_tensor<T>("b", 2, 3);
    auto C0 = create_tensor<T>("C0", 3, 3);
    zero(C0);

    // Perform basic matrix multiplication
    auto C = einsums::linear_algebra::gemm<false, false>(T{1.0}, A, B);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 2; k++) {
                C0(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(C0(i, j) == C(i, j));
        }
    }
}

TEST_CASE("gemm_2") {
    SECTION("double") {
        gemm_test_2<double>();
    }
}

template <typename T>
void syev_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A = create_tensor<T>("a", 3, 3);
    auto b = create_tensor<T>("b", 3);

    A.vector_data() = std::vector<T, einsums::AlignedAllocator<T, 64>>{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};

    // Perform basic matrix multiplication
    einsums::linear_algebra::syev(&A, &b);

    CHECK_THAT(b(0), Catch::Matchers::WithinRel(-0.515729, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinRel(+0.170915, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(+11.344814, 0.00001));
}

TEST_CASE("syev") {
    SECTION("float") {
        syev_test<float>();
    }

    SECTION("double") {
        syev_test<double>();
    }
}

template <typename T>
void heev_test() {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    auto A = create_tensor<T>("a", 3, 3);
    auto b = create_tensor<typename complex_type<T>::type>("b", 3);

    A.vector_data() = std::vector<T, einsums::AlignedAllocator<T, 64>>{
        {0.199889, 0.0},       {-0.330816, -0.127778},  {-0.0546237, 0.176589}, {-0.330816, 0.127778}, {0.629179, 0.0},
        {-0.224813, 0.327171}, {-0.0546237, -0.176589}, {-0.0224813, 0.327171}, {0.170931, 0.0}};

    einsums::linear_algebra::heev(&A, &b);

    // Sometimes 0.0 will be reported as -0.0 therefore we test the Abs of the first two
    CHECK_THAT(b(0), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(1), Catch::Matchers::WithinAbs(0.0, 0.00001));
    CHECK_THAT(b(2), Catch::Matchers::WithinRel(1.0, 0.00001));
}

TEST_CASE("heev") {
    SECTION("float") {
        heev_test<std::complex<float>>();
    }
    SECTION("double") {
        heev_test<std::complex<double>>();
    }
}
