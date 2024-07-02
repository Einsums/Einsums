#pragma once

#include "einsums/_GPUUtils.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"
#include "einsums/utility/IndexUtils.hpp"

#include <bits/utility.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

namespace einsums {
namespace tensor_algebra {

namespace detail {

template <typename CDataType, typename ADataType, typename BDataType, size_t UniqueRank, size_t CRank, size_t ARank, size_t BRank>
__global__ void
einsum_generic_algorithm_gpu(const size_t *unique_strides, const int *C_index_table, const int *A_index_table, const int *B_index_table,
                             const CDataType C_prefactor, CDataType *C, const size_t *C_dims, const size_t *C_stride,
                             const ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const ADataType *A, const size_t *A_dims, const size_t *A_stride, const BDataType *B, const size_t *B_dims,
                             const size_t *B_stride, size_t max_index) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    ssize_t curr_index;

    size_t A_index[ARank], B_index[BRank], C_index[CRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel, C_sentinel;

    curr_index = thread_id;

    // First, set C.
    if (is_zero(C_prefactor)) {
        while (curr_index < C_dims[0] * C_stride[0]) {
            make_zero(C[curr_index]);
            curr_index += kernel_size;
        }
    } else {
        while (curr_index < C_dims[0] * C_stride[0]) {
            C[curr_index] *= C_prefactor;
            curr_index += kernel_size;
        }
    }

    __syncthreads();

    curr_index = thread_id;

    // Now, contract.
    while (curr_index < max_index) {
        sentinel_to_indices<UniqueRank>(curr_index, unique_strides, Unique_index);
        A_sentinel = 0;
        B_sentinel = 0;
        C_sentinel = 0;

        // Unroll these loops since they are known.
#pragma unroll
        for (ssize_t i = 0; i < CRank; i++) {
            C_sentinel += C_stride[i] * Unique_index[C_index_table[i]];
        }

#pragma unroll
        for (ssize_t i = 0; i < ARank; i++) {
            A_sentinel += A_stride[i] * Unique_index[A_index_table[i]];
        }

#pragma unroll
        for (ssize_t i = 0; i < BRank; i++) {
            B_sentinel += B_stride[i] * Unique_index[B_index_table[i]];
        }

        einsums::gpu::atomicAdd_wrap(C + C_sentinel, (CDataType)(AB_prefactor * A[A_sentinel] * B[B_sentinel]));

        curr_index += kernel_size;
    }
}

/**
 * Compute kernel that runs when C has a rank of zero. There are some optimizations that can be made in this case.
 */
template <typename CDataType, typename ADataType, typename BDataType, size_t UniqueRank, size_t ARank, size_t BRank>
__global__ void
einsum_generic_zero_rank_gpu(const size_t *unique_strides, const int *A_index_table, const int *B_index_table, CDataType *C,
                             const ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const ADataType *A, const size_t *A_dims, const size_t *A_stride, const BDataType *B, const size_t *B_dims,
                             const size_t *B_stride, size_t max_index) {

    // Allocated by caller.
    extern __shared__ CDataType work[];

    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    // Clear the work array.
    make_zero(work[thread_id]);

    ssize_t curr_index;

    size_t A_index[ARank], B_index[BRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel;

    curr_index = thread_id;

    while (curr_index < max_index) {
        sentinel_to_indices<UniqueRank>(curr_index, unique_strides, Unique_index);
        A_sentinel = 0;
        B_sentinel = 0;

#pragma unroll
        for (ssize_t i = 0; i < ARank; i++) {
            A_sentinel += A_stride[i] * Unique_index[A_index_table[i]];
        }

#pragma unroll
        for (ssize_t i = 0; i < BRank; i++) {
            B_sentinel += B_stride[i] * Unique_index[B_index_table[i]];
        }

        work[thread_id] += A[A_sentinel] * B[B_sentinel];
    }

    einsums::gpu::atomicAdd_wrap(C, AB_prefactor * work[thread_id]);
}

template <typename... CUniqueIndices, typename... AUniqueIndices, typename... BUniqueIndices, typename... LinkUniqueIndices,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims, typename... LinkDims,
          typename... TargetPositionInC, typename... LinkPositionInLink, template <typename, size_t> typename CType, typename CDataType,
          size_t CRank, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank>
    requires requires {
        requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires RankBasicTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBasicTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
    }
void einsum_generic_algorithm(const std::tuple<CUniqueIndices...> &C_unique, const std::tuple<AUniqueIndices...> & /*A_unique*/,
                              const std::tuple<BUniqueIndices...> & /*B_unique*/, const std::tuple<LinkUniqueIndices...> &link_unique,
                              const std::tuple<CIndices...> & C_indices, const std::tuple<AIndices...> & A_indices,
                              const std::tuple<BIndices...> & B_indices, const std::tuple<TargetDims...> &target_dims,
                              const std::tuple<LinkDims...> &link_dims, const std::tuple<TargetPositionInC...> &target_position_in_C,
                              const std::tuple<LinkPositionInLink...> &link_position_in_link, const CDataType C_prefactor,
                              CType<CDataType, CRank>                                                                *C,
                              const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                              const AType<ADataType, ARank> &A, const BType<BDataType, BRank> &B) {
    using namespace einsums::gpu;

    constexpr auto unique_indices = unique_t<std::tuple<CIndices..., AIndices..., BIndices...>>();
    auto unique_dims = get_dim_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_dims, unique_strides);

    int A_index_table[sizeof...(AIndices)], B_index_table[sizeof...(BIndices)], C_index_table[sizeof...(CIndices)];

    __device_ptr__ int    *A_index_table_gpu, *B_index_table_gpu, *C_index_table_gpu;
    __device_ptr__ size_t *unique_strides_gpu;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    hip_catch(hipMallocAsync((void **)&A_index_table_gpu, sizeof...(AIndices) * sizeof(int), get_stream()));
    hip_catch(hipMallocAsync((void **)&B_index_table_gpu, sizeof...(BIndices) * sizeof(int), get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipMallocAsync((void **)&C_index_table_gpu, sizeof...(CIndices) * sizeof(int), get_stream()));
    }
    hip_catch(hipMallocAsync((void **)&unique_strides_gpu, std::tuple_size<decltype(unique_indices)>::value * sizeof(size_t), get_stream()));

    hip_catch(hipMemcpyAsync((void *)A_index_table_gpu, (const void *)A_index_table, sizeof...(AIndices) * sizeof(int),
                             hipMemcpyHostToDevice, get_stream()));
    hip_catch(hipMemcpyAsync((void *)B_index_table_gpu, (const void *)B_index_table, sizeof...(BIndices) * sizeof(int),
                             hipMemcpyHostToDevice, get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipMemcpyAsync((void *)C_index_table_gpu, (const void *)C_index_table, sizeof...(CIndices) * sizeof(int),
                                 hipMemcpyHostToDevice, get_stream()));
    }
    hip_catch(hipMemcpyAsync((void *)unique_strides_gpu, (const void *)unique_strides.data(), std::tuple_size<decltype(unique_indices)>::value * sizeof(size_t),
                             hipMemcpyHostToDevice, get_stream()));

    // Calculate the optimal launch bounds.
    dim3 threads = block_size(::std::get<0>(unique_dims) * unique_strides[0]),
         grid    = blocks(::std::get<0>(unique_dims) * unique_strides[0]);

    if constexpr (sizeof...(CIndices) != 0) {
        using C_devtype = std::remove_pointer_t<std::decay_t<decltype(C->data())>>;
        using A_devtype = std::remove_pointer_t<std::decay_t<decltype(A.data())>>;
        using B_devtype = std::remove_pointer_t<std::decay_t<decltype(B.data())>>;
        using AB_devtype = std::remove_pointer_t<std::decay_t<std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), decltype(A.data()), decltype(B.data())>>>;

        einsum_generic_algorithm_gpu<C_devtype, A_devtype, B_devtype, std::tuple_size<decltype(unique_indices)>::value, CRank, ARank, BRank>
            <<<threads, grid, 0, get_stream()>>>(unique_strides_gpu, C_index_table_gpu, A_index_table_gpu, B_index_table_gpu, HipCast<CDataType, C_devtype>::cast(C_prefactor),
                                                 C->data(), C->gpu_dims(), C->gpu_strides(), HipCast<decltype(AB_prefactor), AB_devtype>::cast(AB_prefactor), A.data(), A.gpu_dims(),
                                                 A.gpu_strides(), B.data(), B.gpu_dims(), B.gpu_strides(),
                                                 ::std::get<0>(unique_dims) * unique_strides[0]);
    } else {
        // CDataType *work;
        // hip_catch(hipMalloc((void **)&work, threads.x * threads.y * threads.z * blocks.x * blocks.y * blocks.z * sizeof(CDataType)));
        if (C_prefactor == CDataType{0}) {
            *C = CDataType{0};
        } else {
            *C *= C_prefactor;
        }

        using C_devtype = std::remove_pointer_t<std::decay_t<decltype(C->data())>>;
        using A_devtype = std::remove_pointer_t<std::decay_t<decltype(A.data())>>;
        using B_devtype = std::remove_pointer_t<std::decay_t<decltype(B.data())>>;
        using AB_devtype = std::remove_pointer_t<std::decay_t<std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), decltype(A.data()), decltype(B.data())>>>;

        einsum_generic_zero_rank_gpu<C_devtype, A_devtype, B_devtype, std::tuple_size<decltype(unique_indices)>::value, ARank, BRank>
            <<<threads, grid, threads.x * threads.y * threads.z * grid.x * grid.y * grid.z * sizeof(CDataType), get_stream()>>>(
                unique_strides_gpu, A_index_table_gpu, B_index_table_gpu, C->data(), HipCast<decltype(AB_prefactor), AB_devtype>::cast(AB_prefactor), A.data(), A.gpu_dims(), A.gpu_strides(),
                B.data(), B.gpu_dims(), B.gpu_strides(), ::std::get<0>(unique_dims) * unique_strides[0]);
    }

    hip_catch(hipFreeAsync(A_index_table_gpu, get_stream()));
    hip_catch(hipFreeAsync(B_index_table_gpu, get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipFreeAsync(C_index_table_gpu, get_stream()));
    }
    hip_catch(hipFreeAsync(unique_strides_gpu, get_stream()));

    gpu::stream_wait();
}

} // namespace detail

} // namespace tensor_algebra
} // namespace einsums