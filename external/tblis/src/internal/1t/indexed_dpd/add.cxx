#include "util.hpp"
#include "add.hpp"
#include "set.hpp"
#include "scale.hpp"
#include "internal/1t/dense/add.hpp"

#include "external/stl_ext/include/iostream.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add_full(const communicator& comm, const config& cfg,
              T alpha, bool conj_A, const indexed_dpd_marray_view<const T>& A,
              const dim_vector& idx_A_A,
              const dim_vector& idx_A_AB,
                                    const indexed_dpd_marray_view<      T>& B,
              const dim_vector& idx_B_B,
              const dim_vector& idx_B_AB)
{
    marray<T> A2, B2;

    comm.broadcast(
    [&](marray<T>& A2, marray<T>& B2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);

        auto len_A = stl_ext::select_from(A2.lengths(), idx_A_A);
        auto len_B = stl_ext::select_from(B2.lengths(), idx_B_B);
        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto stride_A_A = stl_ext::select_from(A2.strides(), idx_A_A);
        auto stride_B_B = stl_ext::select_from(B2.strides(), idx_B_B);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);

        add(comm, cfg, len_A, len_B, len_AB,
            alpha, conj_A, A2.data(), stride_A_A, stride_A_AB,
             T(0),  false, B2.data(), stride_B_B, stride_B_AB);

        full_to_block(comm, cfg, B2, B);
    },
    A2, B2);
}

template <typename T>
void trace_block(const communicator& comm, const config& cfg,
                 T alpha, bool conj_A, const indexed_dpd_marray_view<const T>& A,
                 const dim_vector& idx_A_A,
                 const dim_vector& idx_A_AB,
                                       const indexed_dpd_marray_view<      T>& B,
                 const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<1> group_A(A, idx_A_A);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_A, irreps_A);

    unsigned irrep_AB = B.irrep();
    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

    unsigned irrep_A = A.irrep()^B.irrep();
    for (auto irrep : group_A.batch_irrep) irrep_A ^= irrep;

    if (group_A.dense_ndim == 0 && irrep_A != 0) return;
    if (group_AB.dense_ndim == 0 && irrep_AB != 0) return;

    group_indices<T, 2> indices_A(A, group_AB, 0, group_A, 0);
    group_indices<T, 1> indices_B(B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];

    stride_type task = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(nidx_B*group_AB.dense_nblock,
                           group_AB.dense_size*group_A.dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<true, false>(idx_A, nidx_A, indices_A, 0,
                                   idx_B, nidx_B, indices_B, 0,
        [&](stride_type next_A)
        {
            if (indices_B[idx_B].factor == T(0)) return;

            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_A,idx_B,block_AB,next_A](const communicator& subcomm)
                {
                    auto local_irreps_A = irreps_A;
                    auto local_irreps_B = irreps_B;

                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                  local_irreps_A, group_AB.dense_idx[0],
                                  local_irreps_B, group_AB.dense_idx[1]);

                    if (is_block_empty(dpd_B, local_irreps_B)) return;

                    auto local_B = dpd_B(local_irreps_B);

                    for (stride_type block_A = 0;block_A < group_A.dense_nblock;block_A++)
                    {
                        assign_irreps(group_A.dense_ndim, irrep_A, nirrep, block_A,
                                      local_irreps_A, group_A.dense_idx[0]);

                        if (is_block_empty(dpd_A, local_irreps_A)) continue;

                        auto local_A = dpd_A(local_irreps_A);

                        len_vector len_AB;
                        stride_vector stride_A_AB, stride_B_AB;
                        stride_type off_A_AB, off_B_AB;
                        get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                           local_A, stride_A_AB, 0,
                                           local_B, stride_B_AB, 1);
                        get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                         local_A, off_A_AB, 0,
                                         local_B, off_B_AB, 1);

                        len_vector len_A;
                        stride_vector stride_A_A;
                        get_local_geometry(indices_A[idx_A].idx[1], group_A, len_A,
                                           local_A, stride_A_A, 0);

                        auto data_B = local_B.data() + indices_B[idx_B].offset + off_B_AB;

                        for (auto local_idx_A = idx_A;local_idx_A < next_A;local_idx_A++)
                        {
                            auto factor = alpha*indices_A[local_idx_A].factor*
                                                indices_B[idx_B].factor;
                            if (factor == T(0)) continue;

                            stride_type off_A_A;
                            get_local_offset(indices_A[idx_A].idx[1], group_A,
                                             local_A, off_A_A, 0);

                            auto data_A = local_A.data() +
                                indices_A[local_idx_A].offset + off_A_A + off_A_AB;

                            add(subcomm, cfg, len_A, {}, len_AB,
                                 factor, conj_A, data_A, stride_A_A, stride_A_AB,
                                   T(1),  false, data_B, {}, stride_B_AB);
                        }
                    }
                });
            }
        });
    });
}

template <typename T>
void replicate_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const indexed_dpd_marray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                                           const indexed_dpd_marray_view<      T>& B,
                     const dim_vector& idx_B_B,
                     const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<1> group_B(B, idx_B_B);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_B, irreps_B);

    unsigned irrep_AB = A.irrep();
    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

    unsigned irrep_B = A.irrep()^B.irrep();
    for (auto irrep : group_B.batch_irrep) irrep_B ^= irrep;

    if (group_B.dense_ndim == 0 && irrep_B != 0) return;
    if (group_AB.dense_ndim == 0 && irrep_AB != 0) return;

    group_indices<T, 1> indices_A(A, group_AB, 0);
    group_indices<T, 2> indices_B(B, group_AB, 1, group_B, 0);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];

    stride_type task = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(nidx_B*group_AB.dense_nblock*group_B.dense_nblock,
                           group_AB.dense_size*group_B.dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, true>(idx_A, nidx_A, indices_A, 0,
                                   idx_B, nidx_B, indices_B, 0,
        [&](stride_type next_B)
        {
            for (auto local_idx_B = idx_B;local_idx_B < next_B;local_idx_B++)
            {
                auto factor = alpha*indices_A[idx_A].factor*
                                    indices_B[local_idx_B].factor;
                if (factor == T(0)) continue;

                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                {
                    for (stride_type block_B = 0;block_B < group_B.dense_nblock;block_B++)
                    {
                        tasks.visit(task++,
                        [&,factor,idx_A,local_idx_B,block_AB,block_B]
                        (const communicator& subcomm)
                        {
                            auto local_irreps_A = irreps_A;
                            auto local_irreps_B = irreps_B;

                            assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                          local_irreps_A, group_AB.dense_idx[0],
                                          local_irreps_B, group_AB.dense_idx[1]);

                            assign_irreps(group_B.dense_ndim, irrep_B, nirrep, block_B,
                                          local_irreps_B, group_B.dense_idx[0]);

                            if (is_block_empty(dpd_A, local_irreps_A)) return;
                            if (is_block_empty(dpd_B, local_irreps_B)) return;

                            auto local_A = dpd_A(local_irreps_A);
                            auto local_B = dpd_B(local_irreps_B);

                            len_vector len_AB;
                            stride_vector stride_A_AB, stride_B_AB;
                            stride_type off_A_AB, off_B_AB;
                            get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                               local_A, stride_A_AB, 0,
                                               local_B, stride_B_AB, 1);
                            get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                             local_A, off_A_AB, 0,
                                             local_B, off_B_AB, 1);

                            len_vector len_B;
                            stride_vector stride_B_B;
                            stride_type off_B_B;
                            get_local_geometry(indices_B[local_idx_B].idx[1], group_B, len_B,
                                               local_B, stride_B_B, 0);
                            get_local_offset(indices_B[local_idx_B].idx[1], group_B,
                                             local_B, off_B_B, 0);

                            auto data_A = local_A.data() + indices_A[idx_A].offset + off_A_AB;
                            auto data_B = local_B.data() + indices_B[local_idx_B].offset + off_B_B + off_B_AB;

                            add(subcomm, cfg, {}, len_B, len_AB,
                                factor, conj_A, data_A, {}, stride_A_AB,
                                  T(1),  false, data_B, stride_B_B, stride_B_AB);
                        });
                    }
                }
            }
        });
    });
}

template <typename T>
void transpose_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const indexed_dpd_marray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                                           const indexed_dpd_marray_view<      T>& B,
                     const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);

    unsigned irrep_AB = A.irrep();
    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

    if (group_AB.dense_ndim == 0 && irrep_AB != 0) return;

    group_indices<T, 1> indices_A(A, group_AB, 0);
    group_indices<T, 1> indices_B(B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];

    stride_type task = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(nidx_B*group_AB.dense_nblock,
                           group_AB.dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, false>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&]
        {
            auto factor = alpha*indices_A[idx_A].factor*indices_B[idx_B].factor;
            if (factor == T(0)) return;

            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                tasks.visit(task++,
                [&,factor,idx_A,idx_B,block_AB](const communicator& subcomm)
                {
                    auto local_irreps_A = irreps_A;
                    auto local_irreps_B = irreps_B;

                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                  local_irreps_A, group_AB.dense_idx[0],
                                  local_irreps_B, group_AB.dense_idx[1]);

                    if (is_block_empty(dpd_A, local_irreps_A)) return;

                    auto local_A = dpd_A(local_irreps_A);
                    auto local_B = dpd_B(local_irreps_B);

                    len_vector len_AB;
                    stride_vector stride_A_AB, stride_B_AB;
                    stride_type off_A_AB, off_B_AB;
                    get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                       local_A, stride_A_AB, 0,
                                       local_B, stride_B_AB, 1);
                    get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                     local_A, off_A_AB, 0,
                                     local_B, off_B_AB, 1);

                    auto data_A = local_A.data() + indices_A[idx_A].offset + off_A_AB;
                    auto data_B = local_B.data() + indices_B[idx_B].offset + off_B_AB;

                    add(subcomm, cfg, {}, {}, len_AB,
                        factor, conj_A, data_A, {}, stride_A_AB,
                          T(1),  false, data_B, {}, stride_B_AB);
                });
            }
        });
    });
}

template <typename T>
void add(const communicator& comm, const config& cfg,
         T alpha, bool conj_A, const indexed_dpd_marray_view<const T>& A,
         const dim_vector& idx_A_A,
         const dim_vector& idx_A_AB,
         T  beta, bool conj_B, const indexed_dpd_marray_view<      T>& B,
         const dim_vector& idx_B_B,
         const dim_vector& idx_B_AB)
{
    if (beta == T(0))
    {
        set(comm, cfg, T(0), B, range(B.dimension()));
    }
    else if (beta != T(1) || (is_complex<T>::value && conj_B))
    {
        scale(comm, cfg, beta, conj_B, B, range(B.dimension()));
    }

    for (unsigned i = 0;i < idx_A_AB.size();i++)
    {
        if (idx_A_AB[i] >= A.dense_dimension() &&
            idx_B_AB[i] >= B.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_AB[i] - A.dense_dimension()) !=
                B.indexed_irrep(idx_B_AB[i] - B.dense_dimension())) return;
        }
    }

    if (dpd_impl == FULL)
    {
        add_full(comm, cfg,
                 alpha, conj_A, A, idx_A_A, idx_A_AB,
                                B, idx_B_B, idx_B_AB);
    }
    else if (!idx_A_A.empty())
    {
        trace_block(comm, cfg,
                    alpha, conj_A, A, idx_A_A, idx_A_AB,
                                   B, idx_B_AB);
    }
    else if (!idx_B_B.empty())
    {
        replicate_block(comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                                       B, idx_B_B, idx_B_AB);
    }
    else
    {
        transpose_block(comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                                       B, idx_B_AB);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, \
                  T alpha, bool conj_A, const indexed_dpd_marray_view<const T>& A, \
                  const dim_vector& idx_A, \
                  const dim_vector& idx_A_AB, \
                  T  beta, bool conj_B, const indexed_dpd_marray_view<      T>& B, \
                  const dim_vector& idx_B, \
                  const dim_vector& idx_B_AB);
#include "configs/foreach_type.h"

}
}
