#include "util.hpp"
#include "dot.hpp"
#include "internal/1t/dense/dot.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot_full(const communicator& comm, const config& cfg,
              bool conj_A, const dpd_marray_view<const T>& A,
              const dim_vector& idx_A_AB,
              bool conj_B, const dpd_marray_view<const T>& B,
              const dim_vector& idx_B_AB,
              T& result)
{
    marray<T> A2, B2;

    comm.broadcast(
    [&](marray<T>& A2, marray<T>& B2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);

        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);

        dot(comm, cfg, len_AB,
            conj_A, A2.data(), stride_A_AB,
            conj_B, B2.data(), stride_B_AB,
            result);
    },
    A2, B2);
}

template <typename T>
void dot_block(const communicator& comm, const config& cfg,
               bool conj_A, const dpd_marray_view<const T>& A,
               const dim_vector& idx_A_AB,
               bool conj_B, const dpd_marray_view<const T>& B,
               const dim_vector& idx_B_AB,
               T& result)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim = A.dimension();

    T local_result = T();
    T block_result;

    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB) nblock_AB *= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;

    irrep_vector irreps_A(ndim);
    irrep_vector irreps_B(ndim);

    unsigned irrep = A.irrep();

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim, irrep, nirrep, block_AB,
              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(A, irreps_A)) continue;

        auto local_A = A(irreps_A);
        auto local_B = B(irreps_B);

        auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

        dot<T>(comm, cfg, len_AB,
               conj_A, local_A.data(), stride_A_AB,
               conj_B, local_B.data(), stride_B_AB, block_result);

        local_result += block_result;
    }

    if (comm.master()) result = local_result;
}

template <typename T>
void dot(const communicator& comm, const config& cfg,
         bool conj_A, const dpd_marray_view<const T>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const dpd_marray_view<const T>& B,
         const dim_vector& idx_B_AB,
         T& result)
{
    if (A.irrep() != B.irrep())
    {
        if (comm.master()) result = 0;
        comm.barrier();
        return;
    }

    if (dpd_impl == FULL)
    {
        dot_full(comm, cfg,
                 conj_A, A, idx_A_AB,
                 conj_B, B, idx_B_AB,
                 result);
    }
    else
    {
        dot_block(comm, cfg,
                  conj_A, A, idx_A_AB,
                  conj_B, B, idx_B_AB,
                  result);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, \
                  bool conj_A, const dpd_marray_view<const T>& A, \
                  const dim_vector& idx_A_AB, \
                  bool conj_B, const dpd_marray_view<const T>& B, \
                  const dim_vector& idx_B_AB, \
                  T& result);
#include "configs/foreach_type.h"

}
}
