#include "dot.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/dot.hpp"
#include "internal/1t/dpd/dot.hpp"
#include "internal/1t/indexed/dot.hpp"
#include "internal/1t/indexed_dpd/dot.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_tensor* A, const label_type* idx_A_,
                      const tblis_tensor* B, const label_type* idx_B_,
                      tblis_scalar* result)
{
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == result->type);

    unsigned ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    unsigned ndim_B = B->ndim;
    len_vector len_B;
    stride_vector stride_B;
    label_vector idx_B;
    diagonal(ndim_B, B->len, B->stride, idx_B_, len_B, stride_B, idx_B);

    if (idx_A.empty() || idx_B.empty())
    {
        len_A.push_back(1);
        len_B.push_back(1);
        stride_A.push_back(0);
        stride_B.push_back(0);
        label_type idx = detail::free_idx(idx_A, idx_B);
        idx_A.push_back(idx);
        idx_B.push_back(idx);
    }

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
    TBLIS_ASSERT(len_AB == stl_ext::select_from(len_B, idx_B, idx_AB));
    auto stride_A_AB = stl_ext::select_from(stride_A, idx_A, idx_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B, idx_B, idx_AB);

    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());

    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(
        [&](const communicator& comm)
        {
            internal::dot<T>(comm, get_config(cfg), len_AB,
                             A->conj, static_cast<T*>(A->data), stride_A_AB,
                             B->conj, static_cast<T*>(B->data), stride_B_AB,
                             result->get<T>());
        }, comm);

        result->get<T>() *= A->alpha<T>()*B->alpha<T>();
    })
}

}

template <typename T>
void dot(const communicator& comm,
         dpd_marray_view<const T> A, const label_type* idx_A_,
         dpd_marray_view<const T> B, const label_type* idx_B_, T& result)
{
    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);

    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                         B.length(idx_B_AB[i], irrep));
        }
    }

    internal::dot<T>(comm, get_default_config(),
                     false, A, idx_A_AB,
                     false, B, idx_B_AB, result);
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, \
                  dpd_marray_view<const T> A, const label_type* idx_A, \
                  dpd_marray_view<const T> B, const label_type* idx_B, T& result);
#include "configs/foreach_type.h"

template <typename T>
void dot(const communicator& comm,
         indexed_marray_view<const T> A, const label_type* idx_A_,
         indexed_marray_view<const T> B, const label_type* idx_B_, T& result)
{
    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);

    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i]) ==
                     B.length(idx_B_AB[i]));
    }

    internal::dot<T>(comm, get_default_config(),
                     false, A, idx_A_AB,
                     false, B, idx_B_AB, result);
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, \
                  indexed_marray_view<const T> A, const label_type* idx_A, \
                  indexed_marray_view<const T> B, const label_type* idx_B, T& result);
#include "configs/foreach_type.h"

template <typename T>
void dot(const communicator& comm,
         indexed_dpd_marray_view<const T> A, const label_type* idx_A_,
         indexed_dpd_marray_view<const T> B, const label_type* idx_B_, T& result)
{
    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);

    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                         B.length(idx_B_AB[i], irrep));
        }
    }

    internal::dot<T>(comm, get_default_config(),
                     false, A, idx_A_AB,
                     false, B, idx_B_AB, result);
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, \
                  indexed_dpd_marray_view<const T> A, const label_type* idx_A, \
                  indexed_dpd_marray_view<const T> B, const label_type* idx_B, T& result);
#include "configs/foreach_type.h"

}
