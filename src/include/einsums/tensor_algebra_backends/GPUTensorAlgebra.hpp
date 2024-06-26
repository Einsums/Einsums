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

__device__ inline bool is_zero(double value) {
    return value == 0.0;
}

__device__ inline bool is_zero(float value) {
    return value == 0.0f;
}

__device__ inline bool is_zero(hipComplex value) {
    return value.x == 0.0f && value.y == 0.0f;
}

__device__ inline bool is_zero(hipDoubleComplex value) {
    return value.x == 0.0 && value.y == 0.0;
}

__device__ inline void make_zero(double &value) {
    value = 0.0;
}

__device__ inline void make_zero(float &value) {
    value = 0.0f;
}

__device__ inline void make_zero(hipComplex &value) {
    value.x = 0.0f;
    value.y = 0.0f;
}

__device__ inline void make_zero(hipDoubleComplex &value) {
    value.x = 0.0;
    value.y = 0.0;
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(float *address, float value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(double *address, double value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(hipComplex *address, hipComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(hipDoubleComplex *address, hipDoubleComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

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

        atomicAdd_wrap(C + C_sentinel, (CDataType)(AB_prefactor * A[A_sentinel] * B[B_sentinel]));

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

    atomicAdd_wrap(C, AB_prefactor * work[thread_id]);
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
        einsum_generic_algorithm_gpu<CDataType, ADataType, BDataType, std::tuple_size<decltype(unique_indices)>::value, CRank, ARank, BRank>
            <<<threads, grid, 0, get_stream()>>>(unique_strides_gpu, C_index_table_gpu, A_index_table_gpu, B_index_table_gpu, C_prefactor,
                                                 C->data(), C->gpu_dims(), C->gpu_strides(), AB_prefactor, A.data(), A.gpu_dims(),
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
        einsum_generic_zero_rank_gpu<CDataType, ADataType, BDataType, std::tuple_size<decltype(unique_indices)>::value, ARank, BRank>
            <<<threads, grid, threads.x * threads.y * threads.z * grid.x * grid.y * grid.z * sizeof(CDataType), get_stream()>>>(
                unique_strides_gpu, A_index_table_gpu, B_index_table_gpu, C->data(), AB_prefactor, A.data(), A.gpu_dims(), A.gpu_strides(),
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

/** Computes the Khatri-Rao product of tensors A and B.
 *
 * Example:
 *    Tensor<2> result = khatri_rao(Indices{I, r}, A, Indices{J, r}, B);
 *
 * Result is described as {(I,J), r}. If multiple common indices are provided they will be collapsed into a single index in the result.
 */
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank,
          typename... AIndices, typename... BIndices, typename T = double>
auto khatri_rao(const ::std::tuple<AIndices...> &, const AType<T, ARank> &A, const ::std::tuple<BIndices...> &, const BType<T, BRank> &B)
    -> ::std::enable_if_t<::std::is_base_of_v<::einsums::detail::TensorBase<T, ARank>, AType<T, ARank>>
                              && ::std::is_base_of_v<::einsums::detail::TensorBase<T, BRank>, BType<T, BRank>>,
                          DeviceTensor<T, 2>> {
    LabeledSection0();

    constexpr auto A_indices = ::std::tuple<AIndices...>();
    constexpr auto B_indices = ::std::tuple<BIndices...>();

    // Determine the common indices between A and B
    constexpr auto common = intersect_t<::std::tuple<AIndices...>, ::std::tuple<BIndices...>>();
    // Determine unique indices in A
    constexpr auto A_only = difference_t<::std::tuple<AIndices...>, decltype(common)>();
    // Determine unique indices in B
    constexpr auto B_only = difference_t<::std::tuple<BIndices...>, decltype(common)>();

    // Record the positions of each types.
    constexpr auto A_common_position = ::einsums::tensor_algebra::detail::find_type_with_position(common, A_indices);
    constexpr auto B_common_position = ::einsums::tensor_algebra::detail::find_type_with_position(common, B_indices);
    constexpr auto A_only_position   = ::einsums::tensor_algebra::detail::find_type_with_position(A_only, A_indices);
    constexpr auto B_only_position   = ::einsums::tensor_algebra::detail::find_type_with_position(B_only, B_indices);

    // Obtain dimensions of the indices discovered above
    auto A_common_dims = ::einsums::tensor_algebra::detail::get_dim_for(A, A_common_position);
    auto B_common_dims = ::einsums::tensor_algebra::detail::get_dim_for(B, B_common_position);
    auto A_only_dims   = ::einsums::tensor_algebra::detail::get_dim_for(A, A_only_position);
    auto B_only_dims   = ::einsums::tensor_algebra::detail::get_dim_for(B, B_only_position);

    // Sanity check - ensure the common dims between A and B are the same size.
    for_sequence<::std::tuple_size_v<decltype(common)>>([&](auto i) {
        if (::std::get<i>(A_common_dims) != ::std::get<i>(B_common_dims)) {
            throw ::std::runtime_error(fmt::format("Common dimensions for index {} of A and B do not match.", ::std::get<i>(common)));
        }
    });

    auto result_dims = ::std::tuple_cat(std::make_tuple("KR product"), A_only_dims, B_only_dims, A_common_dims);

    // Construct resulting tensor
    auto result = ::std::make_from_tuple<DeviceTensor<T, ::std::tuple_size_v<decltype(result_dims)> - 1>>(result_dims);

    // Perform the actual Khatri-Rao product using our einsum routine.
    ::einsums::tensor_algebra::einsum(::std::tuple_cat(A_only, B_only, common), &result, ::std::tuple_cat(A_only, common), A,
                                      ::std::tuple_cat(B_only, common), B);

    // Return a reconstruction of the result tensor ... this can be considered as a simple reshape of the tensor.
    return DeviceTensor<T, 2>{::std::move(result), "KR product", -1, ::einsums::tensor_algebra::detail::product_dims(A_common_position, A)};
}

} // namespace tensor_algebra
} // namespace einsums