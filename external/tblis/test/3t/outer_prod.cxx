#include "../test.hpp"

/*
 * Creates a random tensor outer product operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_outer_prod(stride_type N, T&& A, label_vector& idx_A,
                                      T&& B, label_vector& idx_B,
                                      T&& C, label_vector& idx_C)
{
    unsigned ndim_A, ndim_B, ndim_C;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = ndim_A+ndim_B;
    }
    while (ndim_C > 8);

    random_tensors(N,
                   0, 0, 0,
                   0, ndim_A, ndim_B,
                   0,
                   A, idx_A,
                   B, idx_B,
                   C, idx_C);
}

REPLICATED_TEMPLATED_TEST_CASE(outer_prod, R, T, all_types)
{
    marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    random_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    TENSOR_INFO(A);
    TENSOR_INFO(B);
    TENSOR_INFO(C);

    auto neps = prod(C.lengths());

    T scale(10.0*random_unit<T>());

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_outer_prod, R, T, all_types)
{
    dpd_marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);
    DPD_TENSOR_INFO(C);

    auto neps = dpd_marray<T>::size(C.irrep(), C.lengths());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_outer_prod, R, T, all_types)
{
    indexed_marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);
    INDEXED_TENSOR_INFO(C);

    auto neps = prod(C.lengths());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_outer_prod, R, T, all_types)
{
    indexed_dpd_marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);
    INDEXED_DPD_TENSOR_INFO(C);

    auto neps = dpd_marray<T>::size(C.irrep(), C.lengths());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLOCKED", error, scale*neps);
}
