#include "util.hpp"
#include "add.hpp"
#include "scale.hpp"
#include "set.hpp"
#include "internal/1t/dense/add.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add_full(const communicator& comm, const config& cfg,
              T alpha, bool conj_A, const indexed_marray_view<const T>& A,
              const dim_vector& idx_A_A,
              const dim_vector& idx_A_AB,
                                    const indexed_marray_view<      T>& B,
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
                 T alpha, bool conj_A, const indexed_marray_view<const T>& A,
                 const dim_vector& idx_A_A,
                 const dim_vector& idx_A_AB,
                                       const indexed_marray_view<      T>& B,
                 const dim_vector& idx_B_AB)
{
    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<1> group_A(A, idx_A_A);

    group_indices<T, 2> indices_A(A, group_AB, 0, group_A, 0);
    group_indices<T, 1> indices_B(B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(nidx_B, stl_ext::prod(group_AB.dense_len)*
                                   stl_ext::prod(group_A.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<true, false>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&](stride_type next_A)
        {
            if (indices_B[idx_B].factor == T(0)) return;

            tasks.visit(idx++,
            [&,idx_A,idx_B,next_A](const communicator& subcomm)
            {
                stride_type off_A_AB, off_B_AB;
                get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                 off_A_AB, 0, off_B_AB, 1);

                auto data_B = B.data(0) + indices_B[idx_B].offset + off_B_AB;

                for (auto local_idx_A = idx_A;local_idx_A < next_A;local_idx_A++)
                {
                    auto factor = alpha*indices_A[local_idx_A].factor*
                                        indices_B[idx_B].factor;
                    if (factor == T(0)) continue;

                    auto data_A = A.data(0) + indices_A[local_idx_A].offset + off_A_AB;

                    add(subcomm, cfg, group_A.dense_len, {}, group_AB.dense_len,
                        factor, conj_A, data_A, group_A.dense_stride[0],
                                                group_AB.dense_stride[0],
                          T(1),  false, data_B, {}, group_AB.dense_stride[1]);
                }
            });
        });
    });
}

template <typename T>
void replicate_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const indexed_marray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                                           const indexed_marray_view<      T>& B,
                     const dim_vector& idx_B_B,
                     const dim_vector& idx_B_AB)
{
    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<1> group_B(B, idx_B_B);

    group_indices<T, 1> indices_A(A, group_AB, 0);
    group_indices<T, 2> indices_B(B, group_AB, 1, group_B, 0);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    len_vector dense_len_B = group_AB.dense_len + group_B.dense_len;
    stride_vector dense_stride_B = group_AB.dense_stride[1] + group_B.dense_stride[0];

    comm.do_tasks_deferred(nidx_B, stl_ext::prod(group_AB.dense_len)*
                                   stl_ext::prod(group_B.dense_len)*inout_ratio,
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

                tasks.visit(idx++,
                [&,idx_A,local_idx_B,factor](const communicator& subcomm)
                {
                    stride_type off_A_AB, off_B_AB;
                    get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                     off_A_AB, 0, off_B_AB, 1);

                    auto data_A = A.data(0) + indices_A[idx_A].offset + off_A_AB;
                    auto data_B = B.data(0) + indices_B[local_idx_B].offset + off_B_AB;
                    add(subcomm, cfg, {}, group_B.dense_len, group_AB.dense_len,
                        factor, conj_A, data_A, {}, group_AB.dense_stride[0],
                          T(1),  false, data_B, group_B.dense_stride[0],
                                                group_AB.dense_stride[1]);
                });
            }
        });
    });
}

template <typename T>
void transpose_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const indexed_marray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                                           const indexed_marray_view<      T>& B,
                     const dim_vector& idx_B_AB)
{
    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);

    group_indices<T, 1> indices_A(A, group_AB, 0);
    group_indices<T, 1> indices_B(B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(nidx_B, stl_ext::prod(group_AB.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, false>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&]
        {
            auto factor = alpha*indices_A[idx_A].factor*indices_B[idx_B].factor;
            if (factor == T(0)) return;

            tasks.visit(idx++,
            [&,idx_A,idx_B,factor](const communicator& subcomm)
            {
                stride_type off_A_AB, off_B_AB;
                get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                 off_A_AB, 0, off_B_AB, 1);

                auto data_A = A.data(0) + indices_A[idx_A].offset + off_A_AB;
                auto data_B = B.data(0) + indices_B[idx_B].offset + off_B_AB;

                add(subcomm, cfg, {}, {}, group_AB.dense_len,
                    factor, conj_A, data_A, {}, group_AB.dense_stride[0],
                      T(1),  false, data_B, {}, group_AB.dense_stride[1]);
            });
        });
    });
}

template <typename T>
void add(const communicator& comm, const config& cfg,
         T alpha, bool conj_A, const indexed_marray_view<const T>& A,
         const dim_vector& idx_A_A,
         const dim_vector& idx_A_AB,
         T  beta, bool conj_B, const indexed_marray_view<      T>& B,
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
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, \
                  T alpha, bool conj_A, const indexed_marray_view<const T>& A, \
                  const dim_vector& idx_A, \
                  const dim_vector& idx_A_AB, \
                  T  beta, bool conj_B, const indexed_marray_view<      T>& B, \
                  const dim_vector& idx_B, \
                  const dim_vector& idx_B_AB);
#include "configs/foreach_type.h"

}
}
