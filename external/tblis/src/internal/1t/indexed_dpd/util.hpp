#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_UTIL_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_UTIL_HPP_

#include "util/basic_types.h"
#include "internal/1t/dpd/util.hpp"
#include "internal/1t/indexed/util.hpp"
#include "internal/3t/dpd/mult.hpp"
#include "external/stl_ext/include/zip.hpp"

namespace tblis
{
namespace internal
{

template <typename T, typename U>
void block_to_full(const communicator& comm, const config& cfg,
                   const indexed_dpd_marray_view<T>& A, marray<U>& A2)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned dense_ndim_A = A.dense_dimension();
    unsigned idx_ndim_A = A.indexed_dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A({ndim_A, nirrep});
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }

    if (comm.master()) A2.reset(len_A);
    comm.barrier();

    auto dense_stride_A2 = A2.strides();
    dense_stride_A2.resize(dense_ndim_A);

    A[0].for_each_block(
    [&](const marray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto& dense_len_A = local_A.lengths();
        auto& dense_stride_A = local_A.strides();

        for (len_type i = 0;i < A.num_indices();i++)
        {
            auto data_A = local_A.data() + (A.data(i) - A.data(0));
            auto factor_A = A.factor(i);
            auto idx_A = A.indices(i);

            auto data_A2 = A2.data();
            for (unsigned i = 0;i < dense_ndim_A;i++)
                data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);
            for (unsigned i = dense_ndim_A;i < ndim_A;i++)
                data_A2 += (idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)])*A2.stride(i);

            add<U>(comm, cfg, {}, {}, dense_len_A,
                   factor_A, false,  data_A, {},  dense_stride_A,
                          0, false, data_A2, {}, dense_stride_A2);
        }
    });
}

template <typename T, typename U>
void full_to_block(const communicator& comm, const config& cfg,
                   const marray<U>& A2, const indexed_dpd_marray_view<T>& A)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned dense_ndim_A = A.dense_dimension();
    unsigned idx_ndim_A = A.indexed_dimension();

    matrix<len_type> off_A({ndim_A, nirrep});
    for (unsigned i = 0;i < ndim_A;i++)
    {
        len_type off = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = off;
            off += A.length(i, irrep);
        }
    }

    auto dense_stride_A2 = A2.strides();
    dense_stride_A2.resize(dense_ndim_A);

    A[0].for_each_block(
    [&](const marray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto& dense_len_A = local_A.lengths();
        auto& dense_stride_A = local_A.strides();

        for (len_type i = 0;i < A.num_indices();i++)
        {
            auto data_A = local_A.data() + (A.data(i) - A.data(0));
            auto factor_A = A.factor(i);
            auto idx_A = A.indices(i);

            auto data_A2 = A2.data();
            for (unsigned i = 0;i < dense_ndim_A;i++)
                data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);
            for (unsigned i = dense_ndim_A;i < ndim_A;i++)
                data_A2 += (idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)])*A2.stride(i);

            add<U>(comm, cfg, {}, {}, dense_len_A,
                   factor_A, false, data_A2, {}, dense_stride_A2,
                          1, false,  data_A, {},  dense_stride_A);
        }
    });
}

template <unsigned N> struct dpd_index_group;

template <unsigned I, unsigned N>
void assign_dense_idx_helper(unsigned, dpd_index_group<N>&) {}

template <unsigned I, unsigned N, typename T, typename... Args>
void assign_dense_idx_helper(unsigned i, dpd_index_group<N>& group,
                             const indexed_dpd_marray_view<T>& A,
                             const dim_vector& idx_A, const Args&... args)
{
    group.dense_idx[I].push_back(idx_A[i]);
    assign_dense_idx_helper<I+1>(i, group, args...);
}

template <unsigned N, typename T, typename... Args>
void assign_dense_idx(unsigned i, dpd_index_group<N>& group,
                      const indexed_dpd_marray_view<T>& A,
                      const dim_vector& idx_A, const Args&... args)
{
    assign_dense_idx_helper<0>(i, group, A, idx_A, args...);
}

template <unsigned I, unsigned N>
void assign_mixed_or_batch_idx_helper(unsigned, unsigned,
                                      dpd_index_group<N>&) {}

template <unsigned I, unsigned N, typename T, typename... Args>
void assign_mixed_or_batch_idx_helper(unsigned i, unsigned pos,
                                      dpd_index_group<N>& group,
                                      const indexed_dpd_marray_view<T>& A,
                                      const dim_vector& idx_A, const Args&... args)
{

    if (idx_A[i] < A.dense_dimension())
    {
        group.mixed_idx[I].push_back(idx_A[i]);
        group.mixed_pos[I].push_back(pos);
    }
    else
    {
        unsigned idx = idx_A[i] - A.dense_dimension();

        group.batch_idx[I].push_back(idx);
        group.batch_pos[I].push_back(pos);

        TBLIS_ASSERT(group.batch_irrep[pos] == -1 ||
                     group.batch_irrep[pos] == A.indexed_irrep(idx));
        TBLIS_ASSERT(group.batch_len[pos] == -1 ||
                     group.batch_len[pos] == A.indexed_length(idx));
        group.batch_irrep[pos] = A.indexed_irrep(idx);
        group.batch_len[pos] = A.indexed_length(idx);
    }

    assign_mixed_or_batch_idx_helper<I+1>(i, pos, group, args...);
}

template <unsigned N, typename T, typename... Args>
void assign_mixed_or_batch_idx(unsigned i, unsigned pos,
                               dpd_index_group<N>& group,
                               const indexed_dpd_marray_view<T>& A,
                               const dim_vector& idx_A, const Args&... args)
{
    assign_mixed_or_batch_idx_helper<0>(i, pos, group,
                                        A, idx_A, args...);
}

template <unsigned N>
struct dpd_index_group
{
    unsigned dense_ndim = 0;
    unsigned batch_ndim = 0;
    unsigned dense_nblock = 1;
    stride_type dense_size = 0;
    bool pack_3d = false;

    std::array<dim_vector,N> dense_idx;

    std::array<dim_vector,N> mixed_idx;
    std::array<dim_vector,N> mixed_pos;

    len_vector batch_len;
    stride_vector batch_stride;
    irrep_vector batch_irrep;
    std::array<dim_vector,N> batch_idx;
    std::array<dim_vector,N> batch_pos;

    template <size_t... I>
    dim_vector sort_by_stride(const std::array<stride_vector,N>& dense_stride,
                              stl_ext::detail::integer_sequence<size_t, I...>)
    {
        return detail::sort_by_stride(dense_stride[I]...);
    }

    template <typename T, typename... Args>
    dpd_index_group(const indexed_dpd_marray_view<T>& A, const dim_vector& idx_A,
                    const Args&... args)
    {
        unsigned nirrep = A.num_irreps();

        batch_len.resize(idx_A.size(), -1);
        batch_irrep.resize(idx_A.size(), -1);

        for (unsigned i = 0;i < idx_A.size();i++)
        {
            if (is_idx_dense(i, A, idx_A, args...))
            {
                assign_dense_idx(i, *this, A, idx_A, args...);
                dense_ndim++;
            }
            else
            {
                assign_mixed_or_batch_idx(i, batch_ndim,
                                          *this, A, idx_A, args...);
                batch_ndim++;
            }
        }

        batch_len.resize(batch_ndim);
        batch_stride.resize(batch_ndim);
        batch_irrep.resize(batch_ndim);

        if (batch_ndim > 0) batch_stride[0] = 1;
        for (unsigned i = 1;i < batch_ndim;i++)
            batch_stride[i] = batch_stride[i-1]*batch_len[i-1];

        std::array<len_vector,N> dense_len;
        std::array<stride_vector,N> dense_stride;
        dense_total_lengths_and_strides(dense_len, dense_stride,
                                        A, idx_A, args...);

        dense_size = 1;
        for (unsigned i = 0;i < dense_ndim;i++)
        {
            dense_size *= dense_len[0][i];
            dense_nblock *= nirrep;
        }

        if (dense_nblock > 1)
        {
            dense_size = std::max<stride_type>(1, dense_size/nirrep);
            dense_nblock /= nirrep;
        }

        std::array<stride_vector,N> dense_stride_sub;
        for (unsigned i = 0;i < N;i++)
            dense_stride_sub[i] = stl_ext::select_from(dense_stride[i],
                                                       dense_idx[i]);

        auto reorder = sort_by_stride(dense_stride_sub,
                                      stl_ext::detail::static_range<N>{});

        for (unsigned i = 0;i < N;i++)
            stl_ext::permute(dense_idx[i], reorder);

        unsigned unit = 0;
        for (unsigned i = 0;i < N;i++)
        {
            for (unsigned j = 1;j < dense_ndim;j++)
            {
                if (dense_stride[i][reorder[j]] == 1)
                {
                    pack_3d = true;
                    unit = std::max(unit, j);
                    break;
                }
            }
        }

        if (pack_3d)
            for (unsigned i = 0;i < N;i++)
                std::rotate(dense_idx[i].begin()+1, dense_idx[i].begin()+unit, dense_idx[i].end());
    }
};

template <unsigned I, unsigned N>
void assign_irreps_helper(const dpd_index_group<N>&) {}

template <unsigned I, unsigned N, typename... Args>
void assign_irreps_helper(const dpd_index_group<N>& group,
                          irrep_vector& irreps, Args&... args)
{
    for (unsigned j = 0;j < group.mixed_idx[I].size();j++)
    {
        irreps[group.mixed_idx[I][j]] = group.batch_irrep[group.mixed_pos[I][j]];
    }

    assign_irreps_helper<I+1>(group, args...);
}

template <unsigned N, typename... Args>
void assign_irreps(const dpd_index_group<N>& group, Args&... args)
{
    assign_irreps_helper<0>(group, args...);
}

template <unsigned I, unsigned N>
void get_local_geometry_helper(const len_vector& idx,
                               const dpd_index_group<N>& group,
                               len_vector& len) {}

template <unsigned I, unsigned N, typename T, typename... Args>
void get_local_geometry_helper(const len_vector& idx,
                               const dpd_index_group<N>& group,
                               len_vector& len,  const marray_view<T>& local_A,
                               stride_vector& stride,
                               unsigned i, Args&&... args)
{
    if (I == 0)
        len = stl_ext::select_from(local_A.lengths(), group.dense_idx[I]);

    stride = stl_ext::select_from(local_A.strides(), group.dense_idx[I]);

    get_local_geometry_helper<I+1>(idx, group, len, std::forward<Args>(args)...);
}

template <unsigned N, typename... Args>
void get_local_geometry(const len_vector& idx, const dpd_index_group<N>& group,
                        len_vector& len, Args&&... args)
{
    get_local_geometry_helper<0>(idx, group, len, std::forward<Args>(args)...);
}

template <unsigned I, unsigned N>
void get_local_offset_helper(const len_vector& idx,
                             const dpd_index_group<N>& group) {}

template <unsigned I, unsigned N, typename T, typename... Args>
void get_local_offset_helper(const len_vector& idx,
                             const dpd_index_group<N>& group,
                             const T& A, stride_type& off,
                             unsigned i, Args&&... args)
{
    off = 0;
    for (unsigned j = 0;j < group.mixed_idx[i].size();j++)
        off += idx[group.mixed_pos[i][j]]*
            A.stride(group.mixed_idx[i][j]);

    get_local_offset_helper<I+1>(idx, group, std::forward<Args>(args)...);
}

template <unsigned N, typename... Args>
void get_local_offset(const len_vector& idx, const dpd_index_group<N>& group,
                      Args&&... args)
{
    get_local_offset_helper<0>(idx, group, std::forward<Args>(args)...);
}

}
}

#endif
