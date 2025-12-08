//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TypeSupport/GPUCast.hpp>
#include <Einsums/TypeSupport/GPUComplex.hpp>

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
einsum_generic_algorithm_gpu(size_t const *__restrict__ unique_strides, int const *__restrict__ C_index_table,
                             int const *__restrict__ A_index_table, int const *__restrict__ B_index_table, CDataType const C_prefactor,
                             CDataType *__restrict__ C, size_t const *__restrict__ C_dims, size_t const *__restrict__ C_stride,
                             std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> const AB_prefactor,
                             ADataType const *__restrict__ A, size_t const *__restrict__ A_dims, size_t const *__restrict__ A_stride,
                             BDataType const *__restrict__ B, size_t const *__restrict__ B_dims, size_t const *__restrict__ B_stride,
                             size_t max_index) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t A_index[ARank], B_index[BRank], C_index[CRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel, C_sentinel;

    // Now, contract.
    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
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

        einsums::gpu::atomicAdd_wrap(C + C_sentinel, (CDataType)(gpu_ops::mult(gpu_ops::mult(AB_prefactor, A[A_sentinel]), B[B_sentinel])));
    }
}

// When we will only see a certain element once, we can ignore atomicity for a speedup.
template <typename CDataType, typename ADataType, typename BDataType, size_t UniqueRank, size_t CRank, size_t ARank, size_t BRank>
__global__ void einsum_generic_algorithm_direct_product_gpu(
    size_t const *__restrict__ unique_strides, int const *__restrict__ C_index_table, int const *__restrict__ A_index_table,
    int const *__restrict__ B_index_table, CDataType const C_prefactor, CDataType *__restrict__ C, size_t const *__restrict__ C_dims,
    size_t const *__restrict__ C_stride,
    std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> const AB_prefactor, ADataType const *__restrict__ A,
    size_t const *__restrict__ A_dims, size_t const *__restrict__ A_stride, BDataType const *__restrict__ B,
    size_t const *__restrict__ B_dims, size_t const *__restrict__ B_stride, size_t max_index) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t A_index[ARank], B_index[BRank], C_index[CRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel, C_sentinel;

    // Now, contract.
    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
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

        C[C_sentinel] = gpu_ops::fma(gpu_ops::mult(AB_prefactor, A[A_sentinel]), B[B_sentinel], C[C_sentinel]);
    }
}

/**
 * Compute kernel that runs when C has a rank of zero. There are some optimizations that can be made in this case.
 */
template <typename CDataType, typename ADataType, typename BDataType, size_t UniqueRank, size_t ARank, size_t BRank>
__global__ void einsum_generic_zero_rank_gpu(
    size_t const *__restrict__ unique_strides, int const *__restrict__ A_index_table, int const *__restrict__ B_index_table,
    CDataType *__restrict__ C, std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> const AB_prefactor,
    ADataType const *__restrict__ A, size_t const *__restrict__ A_dims, size_t const *__restrict__ A_stride,
    BDataType const *__restrict__ B, size_t const *__restrict__ B_dims, size_t const *__restrict__ B_stride, size_t max_index) {

    CDataType value;

    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    // Clear the dot product.
    make_zero(value);

    __syncthreads();

    size_t A_index[ARank], B_index[BRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel;

    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
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

        value = gpu_ops::fma(A[A_sentinel], B[B_sentinel], value);
    }

    atomicAdd_wrap(C, gpu_ops::mult(AB_prefactor, value));
}

template <typename... CUniqueIndices, typename... AUniqueIndices, typename... BUniqueIndices, typename... LinkUniqueIndices,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims, typename... LinkDims,
          typename... TargetPositionInC, typename... LinkPositionInLink, typename CType, DeviceBasicTensorConcept AType,
          DeviceBasicTensorConcept BType>
    requires(DeviceBasicTensorConcept<CType> || (!TensorConcept<CType> && sizeof...(CIndices) == 0))
void einsum_generic_algorithm(std::tuple<CUniqueIndices...> const &C_unique, std::tuple<AUniqueIndices...> const & /*A_unique*/,
                              std::tuple<BUniqueIndices...> const & /*B_unique*/, std::tuple<LinkUniqueIndices...> const &link_unique,
                              std::tuple<CIndices...> const &C_indices, std::tuple<AIndices...> const &A_indices,
                              std::tuple<BIndices...> const &B_indices, std::tuple<TargetDims...> const &target_dims,
                              std::tuple<LinkDims...> const &link_dims, std::tuple<TargetPositionInC...> const &target_position_in_C,
                              std::tuple<LinkPositionInLink...> const &link_position_in_link, ValueTypeT<CType> const C_prefactor, CType *C,
                              BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor, AType const &A,
                              BType const &B) {
    using namespace einsums::gpu;

    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr bool direct_product_swap =
        (sizeof...(AIndices) == sizeof...(BIndices)) && (sizeof...(AIndices) == sizeof...(CIndices)) &&
        (std::tuple_size_v<IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>> == sizeof...(AIndices)) &&
        (std::tuple_size_v<IntersectT<std::tuple<AIndices...>, std::tuple<CIndices...>>> == sizeof...(AIndices)) &&
        (std::tuple_size_v<IntersectT<std::tuple<CIndices...>, std::tuple<BIndices...>>> == sizeof...(AIndices));

    constexpr auto unique_indices = UniqueT<std::tuple<CIndices..., AIndices..., BIndices...>>();
    auto           unique_dims    = get_dim_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_dims, unique_strides);

    int A_index_table[sizeof...(AIndices)], B_index_table[sizeof...(BIndices)], C_index_table[sizeof...(CIndices)];

    int    *A_index_table_gpu, *B_index_table_gpu, *C_index_table_gpu;
    size_t *unique_strides_gpu;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    hip_catch(hipMallocAsync((void **)&A_index_table_gpu, sizeof...(AIndices) * sizeof(int), get_stream()));
    hip_catch(hipMallocAsync((void **)&B_index_table_gpu, sizeof...(BIndices) * sizeof(int), get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipMallocAsync((void **)&C_index_table_gpu, sizeof...(CIndices) * sizeof(int), get_stream()));
    }
    hip_catch(
        hipMallocAsync((void **)&unique_strides_gpu, std::tuple_size<decltype(unique_indices)>::value * sizeof(size_t), get_stream()));

    hip_catch(hipMemcpyAsync((void *)A_index_table_gpu, (void const *)A_index_table, sizeof...(AIndices) * sizeof(int),
                             hipMemcpyHostToDevice, get_stream()));
    hip_catch(hipMemcpyAsync((void *)B_index_table_gpu, (void const *)B_index_table, sizeof...(BIndices) * sizeof(int),
                             hipMemcpyHostToDevice, get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipMemcpyAsync((void *)C_index_table_gpu, (void const *)C_index_table, sizeof...(CIndices) * sizeof(int),
                                 hipMemcpyHostToDevice, get_stream()));
    }
    hip_catch(hipMemcpyAsync((void *)unique_strides_gpu, (void const *)unique_strides.data(),
                             std::tuple_size<decltype(unique_indices)>::value * sizeof(size_t), hipMemcpyHostToDevice, get_stream()));

    // Calculate the optimal launch bounds.
    dim3 threads = block_size(std::get<0>(unique_dims) * unique_strides[0]), grid = blocks(std::get<0>(unique_dims) * unique_strides[0]);

    if constexpr (sizeof...(CIndices) != 0) {
        using C_devtype   = typename CType::dev_datatype;
        using A_devtype   = typename AType::dev_datatype;
        using B_devtype   = typename BType::dev_datatype;
        using AB_devtype  = BiggestTypeT<A_devtype, B_devtype>;
        using C_hosttype  = typename CType::host_datatype;
        using A_hosttype  = typename AType::host_datatype;
        using B_hosttype  = typename BType::host_datatype;
        using AB_hosttype = BiggestTypeT<A_hosttype, B_hosttype>;

        if (C_prefactor == C_hosttype{0.0}) {
            C->zero();
        } else {
            *C *= C_prefactor;
        }

        if constexpr (!direct_product_swap) {
            einsum_generic_algorithm_gpu<C_devtype, A_devtype, B_devtype, std::tuple_size<decltype(unique_indices)>::value, CRank, ARank,
                                         BRank><<<grid, threads, 0, get_stream()>>>(
                unique_strides_gpu, C_index_table_gpu, A_index_table_gpu, B_index_table_gpu,
                HipCast<C_devtype, C_hosttype>::cast(C_prefactor), C->gpu_data(), C->gpu_dims(), C->gpu_strides(),
                HipCast<AB_devtype, AB_hosttype>::cast(AB_prefactor), A.gpu_data(), A.gpu_dims(), A.gpu_strides(), B.gpu_data(),
                B.gpu_dims(), B.gpu_strides(), std::get<0>(unique_dims) * unique_strides[0]);
        } else {
            einsum_generic_algorithm_direct_product_gpu<C_devtype, A_devtype, B_devtype, std::tuple_size<decltype(unique_indices)>::value,
                                                        CRank, ARank, BRank><<<grid, threads, 0, get_stream()>>>(
                unique_strides_gpu, C_index_table_gpu, A_index_table_gpu, B_index_table_gpu,
                HipCast<C_devtype, C_hosttype>::cast(C_prefactor), C->gpu_data(), C->gpu_dims(), C->gpu_strides(),
                HipCast<AB_devtype, AB_hosttype>::cast(AB_prefactor), A.gpu_data(), A.gpu_dims(), A.gpu_strides(), B.gpu_data(),
                B.gpu_dims(), B.gpu_strides(), std::get<0>(unique_dims) * unique_strides[0]);
        }
        gpu::stream_wait();

    } else {
        using C_devtype   = typename einsums::tensor_base::DeviceTypedTensor<ValueTypeT<CType>>::dev_datatype;
        using A_devtype   = typename AType::dev_datatype;
        using B_devtype   = typename BType::dev_datatype;
        using AB_devtype  = BiggestTypeT<A_devtype, B_devtype>;
        using C_hosttype  = ValueTypeT<CType>;
        using A_hosttype  = typename AType::host_datatype;
        using B_hosttype  = typename BType::host_datatype;
        using AB_hosttype = BiggestTypeT<A_hosttype, B_hosttype>;

        // CDataType *work;
        // hip_catch(hipMalloc((void **)&work, threads.x * threads.y * threads.z * blocks.x * blocks.y * blocks.z * sizeof(CDataType)));
        if (C_prefactor == ValueTypeT<CType>{0.0}) {
            *C = ValueTypeT<CType>{0.0};
        } else {
            *C *= C_prefactor;
        }

        C_devtype *C_data;

        if constexpr (einsums::IsTensorV<CType>) {
            C_data = C->gpu_data();
        } else {
            hip_catch(hipMalloc((void **)&C_data, sizeof(C_devtype)));
            hip_catch(hipMemcpyAsync(C_data, C, sizeof(C_devtype), hipMemcpyHostToDevice, get_stream()));
        }

        einsum_generic_zero_rank_gpu<C_devtype, A_devtype, B_devtype, std::tuple_size<decltype(unique_indices)>::value, ARank, BRank>
            <<<grid, threads, 0, get_stream()>>>(unique_strides_gpu, A_index_table_gpu, B_index_table_gpu, C_data,
                                                 HipCast<AB_devtype, AB_hosttype>::cast(AB_prefactor), A.gpu_data(), A.gpu_dims(),
                                                 A.gpu_strides(), B.gpu_data(), B.gpu_dims(), B.gpu_strides(),
                                                 std::get<0>(unique_dims) * unique_strides[0]);
        gpu::stream_wait();

        if constexpr (!einsums::IsTensorV<CType>) {
            hip_catch(hipMemcpy(C, C_data, sizeof(C_devtype), hipMemcpyDeviceToHost));
            // No sync
            hip_catch(hipFree(C_data));
        }
    }

    hip_catch(hipFreeAsync(A_index_table_gpu, get_stream()));
    hip_catch(hipFreeAsync(B_index_table_gpu, get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipFreeAsync(C_index_table_gpu, get_stream()));
    }
    hip_catch(hipFreeAsync(unique_strides_gpu, get_stream()));
}

} // namespace detail

} // namespace tensor_algebra
} // namespace einsums