#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/_Index.hpp"
#include "mkl_cblas.h"

#include <type_traits>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::tensor_algebra::index;

// Playing around with permutational symmetry
struct PermutationalSymmetryBase {};

struct ILessThanJ : PermutationalSymmetryBase {};
struct ILessThanEqualJ : PermutationalSymmetryBase {};
struct IGreaterThanJ : PermutationalSymmetryBase {};
struct IGreaterThanEqualJ : PermutationalSymmetryBase {};

template <typename LHS, typename RHS>
constexpr auto operator<(LHS, RHS) -> std::enable_if_t<std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, LHS> &&
                                                           std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, RHS>,
                                                       ILessThanJ> {
    return {};
}

template <typename LHS, typename RHS>
constexpr auto operator<=(LHS, RHS) -> std::enable_if_t<std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, LHS> &&
                                                            std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, RHS>,
                                                        ILessThanEqualJ> {
    return {};
}

template <typename LHS, typename RHS>
constexpr auto operator>(LHS, RHS) -> std::enable_if_t<std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, LHS> &&
                                                           std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, RHS>,
                                                       IGreaterThanJ> {
    return {};
}

template <typename LHS, typename RHS>
constexpr auto operator>=(LHS, RHS) -> std::enable_if_t<std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, LHS> &&
                                                            std::is_base_of_v<einsums::tensor_algebra::index::LabelBase, RHS>,
                                                        IGreaterThanEqualJ> {
    return {};
}

void test_permutational_symmetry() {
    constexpr auto result1 = i < j;
    constexpr auto result2 = i <= j;
    constexpr auto result3 = i > j;
    constexpr auto result4 = i >= j;
}

namespace test {

template <typename... Args>
struct Dims : public std::tuple<Args...> {
    Dims(Args... args) : std::tuple<Args...>(args...){};
};

template <typename... Args>
struct PS : public std::tuple<Args...> {
    PS(Args... args) : std::tuple<Args...>(args...){};
};

template <typename T, size_t Rank>
struct Tensor : public einsums::detail::TensorBase<T, Rank> {

    template <typename... dims, typename... ps>
    explicit Tensor(std::string name, Dims dims, PS ps) {
        static_assert(2 == sizeof...(PS));
        static_assert(2 == sizeof...(Dims));
    }

    [[nodiscard]] auto dim(int d) const -> size_t override { return 0; }
};

} // namespace test

void test_tensor_construction() {
    auto t = test::Tensor<double, 2>{"test", std::make_tuple(3, 3), std::make_tuple(i, i >= j)};
}

template <template <typename, size_t> typename CType, template <typename, size_t> typename... MultiTensors, size_t Rank,
          typename MultiOperator, typename T = double>
auto element_if(MultiOperator multi_opt, const CType<T, Rank> &A, const CType<T, Rank> &B) {
    // LabeledSection0();

    auto target_dims = get_dim_ranges<Rank>(A);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

    // Ensure the various tensors passed in are the same dimensionality
    if (A.dims() != B.dims()) {
        println_abort("element: tensors A and B do not have same dimensionality");
    }

    bool result{false};
    for (auto it = view.begin(); it != view.end(); it++) {
        const T &A_value = std::apply(A, *it);
        const T &B_value = std::apply(B, *it);
        result |= multi_opt(A_value, B_value);
    }

    return result;
}

// Returns false if they are equal. True if not
template <typename T, size_t Rank>
auto not_equal(const Tensor<T, Rank> &A, const Tensor<T, Rank> &B, T tolerance = 1.0E-10) -> bool {
    return element_if([&](const T &A_value, const T &B_value) -> bool { return std::abs(A_value - B_value) > tolerance; }, A, B);
}

void test1() {
    const int B = 5, C = 5, D = 5;
    const int K = 2, L = 2;

    auto t_temp_vv = create_tensor("t", B, C);
    auto KLCD      = create_tensor("v", K, L, C, D);
    auto t_init    = create_tensor("t", B, D, K, L);

    // T_bc = V_klcd * T_bdkl
    einsum(0.0, Indices{b, c}, &t_temp_vv, -0.5, Indices{k, l, c, d}, KLCD, Indices{b, d, k, l}, t_init);

    // Swapping to
    // T_bc = T_bdkl * V_klcd
    auto t1 = create_tensor("t", 5, 5);
    zero(t1);

    int lda = D, ldb = D, ldc = C;
    int strideA = K * L, strideB = K * L, strideC = 0;
    int BATCH = K * L;

    // Duh of course this won't work.
    cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, B, C, D, 1.0, t_init.vector_data().data(), lda, strideA,
                              KLCD.vector_data().data(), ldb, strideB, 1.0, t1.vector_data().data(), ldc, strideC, BATCH);
}

void test2() {
    Section   test2{"test2"};
    const int A = 5, B = 5, C = 5, I = 2, J = 2;

    auto t_temp_vvoo = create_tensor("t", A, B, I, J);
    auto t_temp_vv   = create_random_tensor("t", B, C);
    auto t_init      = create_random_tensor("t", A, C, I, J);
    auto t1          = create_tensor("t", A, B, I, J);
    auto t2          = create_tensor("t", A, B, I, J);

    for (int counter = 0; counter < 1; counter++) {
        einsum(0.0, Indices{a, b, i, j}, &t_temp_vvoo, 1.0, Indices{b, c}, t_temp_vv, Indices{a, c, i, j}, t_init);

        {
            Section section{"cblas_dgemm_batch_strided A"};

            int M = B, N = I * J, K = C;
            int lda = C, ldb = I * J, ldc = I * J;
            int strideA = 0, strideB = C * I * J, strideC = B * I * J, BATCH = A;

            cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, t_temp_vv.vector_data().data(), lda, strideA,
                                      t_init.vector_data().data(), ldb, strideB, 0.0, t1.vector_data().data(), ldc, strideC, BATCH);
        }

        {
            Section section{"cblas_dgemm_batch_strided B"};

            int M = A, N = B, K = C;
            int lda = C * I * J, ldb = C, ldc = B * I * J;
            int strideA = 1, strideB = 0, strideC = I * J, BATCH = I * J;

            for (int p = 0; p < BATCH; ++p) {
                for (int a = 0; a < A; ++a) {
                    for (int b = 0; b < B; ++b) {
                        double c_mnp = 0.0;
                        for (int c = 0; c < C; ++c) {
                            // abij = acij * bcT
                            // ab(ij) = ac(ij) * bcT
                            c_mnp += t_init.vector_data().data()[a * lda + c + p * strideA] *
                                     t_temp_vv.vector_data().data()[b * ldb + c + p * strideB];
                        }
                        t2.vector_data().data()[a * ldc + b + p * strideC] = c_mnp;
                    }
                }
            }

            // cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, t_init.vector_data().data(), lda, strideA,
            //                           t_temp_vv.vector_data().data(), ldb, strideB, 0.0, t2.vector_data().data(), ldc, strideC, BATCH);
        }
    }

    auto same1 = !not_equal(t_temp_vvoo, t1);
    println("tensors are equal? {}", same1);

    auto same2 = !not_equal(t_temp_vvoo, t2);
    println("tensors are equal? {}", same2);

    // Section sorted{"sorted tensors"};
    // for (int counter = 0; counter < 1; counter++) {
    //     sort(Indices{c, a, i, j}, &t1, Indices{a, c, i, j}, t_init);
    //     einsum(Indices{b, a, i, j}, &t1, Indices{b, c}, t_temp_vv, Indices{c, a, i, j}, t_init);
    //     sort(Indices{a, c, i, j}, &t1, Indices{c, a, i, j}, t_init);
    // }
}

auto main() -> int {
    initialize();

#if 0
    const int M = 10, N = 10, K = 30, BATCH = 20;

    auto A  = create_random_tensor("A", M, BATCH, K);
    auto B  = create_random_tensor("B", K, N);
    auto C  = create_tensor("C", M, BATCH, N);
    auto C1 = create_tensor("C", M, BATCH, N);
    auto C2 = create_tensor("C", M, BATCH, N);

    for (int counter = 0; counter < 1; counter++) {
        zero(C);
        zero(C1);
        zero(C2);

        einsum(Indices{m, b, n}, &C, Indices{m, b, k}, A, Indices{k, n}, B);

        int lda = K, ldb = N, ldc = N;
        int strideA = M * K, strideB = 0, strideC = M * N;

        {
            Section section("cblas_dgemm_batch_strided");
            cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A.vector_data().data(), lda, strideA,
                                      B.vector_data().data(), ldb, strideB, 0.0, C1.vector_data().data(), ldc, strideC, BATCH);
        }

        {
            Section section("for loops");
            for (int p = 0; p < BATCH; ++p) {
                for (int m = 0; m < M; ++m) {
                    for (int n = 0; n < N; ++n) {
                        double c_mnp = 0.0;
                        for (int k = 0; k < K; ++k) {
                            c_mnp += A.vector_data().data()[m * lda + p * strideA + k] * B.vector_data().data()[k * ldb + n + p * strideB];
                        }
                        C2.vector_data().data()[m * ldc + p * strideC + n] = c_mnp;
                    }
                }
            }
        }
    }

    // println(C);
    // println(C1);
    // println(C2);

    auto result = not_equal(C, C1);
    println("tensors are not equal? {}", result);
#endif

    // test2();

    finalize(true);
    return EXIT_SUCCESS;
}