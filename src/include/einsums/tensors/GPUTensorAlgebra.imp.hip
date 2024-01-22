#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/GPUTensorAlgebra.hpp"
#include <bits/utility.h>
#include <type_traits>

namespace einsums {
namespace gpu {
namespace tensor_algebra {

namespace detail {

__host__ __device__ void sentinel_to_indices(size_t sentinel, const size_t *unique_strides, size_t num_unique_inds, size_t *out_inds) {
    size_t hold = sentinel;
    for (ssize_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            out_inds[i] = 0;
        }
    }
}

template <typename CDataType, typename ADataType, typename BDataType, size_t UniqueRank, size_t CRank, size_t ARank, size_t BRank>
__global__ void einsums_generic_algorithm_gpu(
    const size_t *unique_strides, const int *C_index_table, const int *A_index_table,
    const int *B_index_table, const CDataType C_prefactor, CDataType *C, const size_t *C_dims, const size_t *C_stride,
    const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor, const ADataType *A,
    const size_t *A_dims, const size_t *A_stride, const BDataType *B, const size_t *B_dims, const size_t *B_stride, size_t max_index) {

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    ssize_t curr_index;

    size_t A_index[ARank], B_index[BRank], C_index[CRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel, C_sentinel;

    curr_index = thread_id;

    // First, set C.
    while (curr_index < C_dims[0] * C_stride[0]) {
        C[curr_index] *= C_prefactor;
        curr_index += kernel_size;
    }

    __syncthreads();

    // Now, contract.
    while (curr_index < max_index) {
        sentinel_to_indices(curr_index, unique_strides, UniqueRank, Unique_index);
        A_sentinel = 0;
        B_sentinel = 0;
        C_sentinel = 0;

        for (ssize_t i = 0; i < CRank; i++) {
            C_sentinel += C_stride[i] * Unique_index[C_index_table[i]];
        }

        for (ssize_t i = 0; i < ARank; i++) {
            A_sentinel += A_stride[i] * Unique_index[A_index_table[i]];
        }

        for (ssize_t i = 0; i < BRank; i++) {
            B_sentinel += B_stride[i] * Unique_index[B_index_table[i]];
        }

        atomicAdd(C + C_sentinel, (CDataType)(AB_prefactor * A[A_sentinel] * B[B_sentinel]));

        curr_index += kernel_size;
    }
}

template<typename... UniqueDims, size_t... I>
void dims_to_strides(const std::tuple<UniqueDims...> &dims, size_t *out, std::index_sequence<I...>) {
    std::array<size_t, sizeof...(UniqueDims)> arr{std::get<I>(dims)...};

    size_t stride = 1;

    for(int i = sizeof...(UniqueDims) - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= arr[i];
    }
}

template<typename... UniqueDims>
void dims_to_strides(const std::tuple<UniqueDims...> &dims, size_t *out) {
    dims_to_strides(dims, out, std::make_index_sequence<sizeof...(UniqueDims)>());
}

template<typename Head, typename Index, int I>
void compile_index_table(const std::tuple<Head> &, const Index &, int &out) {
    if constexpr(std::is_same_v<Head, Index>) {
        out = I;
    } else {
        out = -1;
    }
}

template<typename Head, typename... UniqueIndices, typename Index, int I>
void compile_index_table(const std::tuple<Head, UniqueIndices...> &, const Index &index, int &out) {
    if constexpr (std::is_same_v<Head, Index>) {
        out = I;
    } else {
        compile_index_table<I + 1>(std::tuple<UniqueIndices...>(), index, out);
    }
}

template<typename... UniqueIndices, typename... Indices, size_t... I>
void compile_index_table(const std::tuple<UniqueIndices...> &from_inds, const std::tuple<Indices...> &to_inds, int *out, std::index_sequence<I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<I>(to_inds), out[I])...};
}

template<typename... UniqueIndices, typename... Indices>
void compile_index_table(const std::tuple<UniqueIndices...> &from_inds, const std::tuple<Indices...> &to_inds, int *out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}

template <typename... UniqueIndices, typename... CIndices, typename... AIndices, typename... BIndices, typename... UniqueDims,
          template <typename, size_t> typename CType, typename CDataType, size_t CRank, template <typename, size_t> typename AType,
          typename ADataType, size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
void einsum_generic_algorithm(const std::tuple<UniqueIndices...> &unique_indices, const std::tuple<CIndices...> &C_indices,
                              const std::tuple<AIndices...> &A_indices, const std::tuple<BIndices...> &B_indices,
                              const std::tuple<UniqueDims...> &unique_dims, const CDataType C_prefactor, CType<CDataType, CRank> *C,
                              const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                              const AType<ADataType, ARank> &A, const BType<BDataType, BRank> &B, dim3 threads = dim3(32),
                              dim3 blocks = dim3(32)) {

    size_t unique_strides[sizeof...(UniqueIndices)];

    dims_to_strides(unique_dims, unique_strides);

    int A_index_table[sizeof...(AIndices)], B_index_table[sizeof...(BIndices)], C_index_table[sizeof...(CIndices)];

    device_ptr int *A_index_table_gpu, *B_index_table_gpu, *C_index_table_gpu;
    device_ptr size_t *unique_strides_gpu;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    hip_catch(hipMalloc((void **) &A_index_table_gpu, sizeof...(AIndices) * sizeof(int)));
    hip_catch(hipMalloc((void **) &B_index_table_gpu, sizeof...(BIndices) * sizeof(int)));
    hip_catch(hipMalloc((void **) &C_index_table_gpu, sizeof...(CIndices) * sizeof(int)));
    hip_catch(hipMalloc((void **) &unique_strides_gpu, sizeof...(UniqueIndices) * sizeof(size_t)));

    hip_catch(hipMemcpy((void *) A_index_table_gpu, (const void *) A_index_table, sizeof...(AIndices) * sizeof(int), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *) B_index_table_gpu, (const void *) B_index_table, sizeof...(BIndices) * sizeof(int), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *) C_index_table_gpu, (const void *) C_index_table, sizeof...(CIndices) * sizeof(int), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *) unique_strides_gpu, (const void *) unique_strides, sizeof...(UniqueIndices) * sizeof(size_t), hipMemcpyHostToDevice));

    einsums_generic_algorithm_gpu<<<threads, blocks>>>(unique_strides_gpu, C_index_table_gpu, A_index_table_gpu,  B_index_table_gpu, C_prefactor, C->data(), C->gpu_dims(), C->gpu_stride(),
        AB_prefactor, A.data(), A.gpu_dims(), A.gpu_stride(), B.data(), B.gpu_dims(), B.gpu_stride(), std::get<0>(unique_dims) * unique_strides[0]);

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipFree((void *) A_index_table_gpu));
    hip_catch(hipFree((void *) B_index_table_gpu));
    hip_catch(hipFree((void *) C_index_table_gpu));
    hip_catch(hipFree((void *) unique_strides_gpu));

}

} // namespace detail

} // namespace tensor_algebra
} // namespace gpu
} // namespace einsums