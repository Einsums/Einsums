//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_GPUCast.hpp"
#include "einsums/_GPUUtils.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace einsums::tensor_algebra {

namespace detail {

#if defined(EINSUMS_USE_HPTT)

void EINSUMS_EXPORT gpu_sort(const int *perm, const int dim, const float alpha, const float *A, const int *sizeA, const float beta,
                             float *B);
void EINSUMS_EXPORT gpu_sort(const int *perm, const int dim, const double alpha, const double *A, const int *sizeA, const double beta,
                             double *B);
void EINSUMS_EXPORT gpu_sort(const int *perm, const int dim, const hipComplex alpha, const hipComplex *A, const int *sizeA,
                             const hipComplex beta, hipComplex *B);
void EINSUMS_EXPORT gpu_sort(const int *perm, const int dim, const hipDoubleComplex alpha, const hipDoubleComplex *A, const int *sizeA,
                             const hipDoubleComplex beta, hipDoubleComplex *B);
#endif

template <typename T, size_t Rank>
__global__ void sort_kernel(const int *perm, const T alpha, const T *A, const size_t *strideA, const T beta, T *B, const size_t *strideB,
                            size_t size) {
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t A_index[Rank], B_index[Rank];
    size_t A_sentinel, B_sentinel;

    for (ssize_t curr_index = thread_id; curr_index < size; curr_index++) {
        einsums::tensor_algebra::detail::sentinel_to_indices<Rank>(curr_index, strideA, A_index);
        A_sentinel = 0;
        B_sentinel = 0;

#pragma unroll
        for (ssize_t i = 0; i < Rank; i++) {
            A_sentinel += strideA[i] * A_index[i];
        }

#pragma unroll
        for (ssize_t i = 0; i < Rank; i++) {
            B_sentinel += strideB[i] * A_index[perm[i]];
        }

        B[B_sentinel] = beta * B[B_sentinel] + alpha * A[A_sentinel];
    }
}

template <typename Index>
auto reverse_inds(const std::tuple<Index> &ind) {
    return ind;
}

template <typename Head, typename... Tail>
    requires(sizeof...(Tail) > 0)
auto reverse_inds(const std::tuple<Head, Tail...> &inds) {
    return std::tuple_cat(reverse_inds(std::tuple<Tail...>{}), std::tuple<Head>{});
}

} // namespace detail

//
// sort algorithm
//
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename CType, size_t CRank,
          typename... CIndices, typename... AIndices, typename U, typename T = double>
    requires requires {
        requires DeviceRankTensor<AType<T, ARank>, ARank, T>;
        requires DeviceRankTensor<CType<T, CRank>, CRank, T>;
    }
auto sort(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<T, CRank> *C, const U UA_prefactor,
          const std::tuple<AIndices...> &A_indices,
          const AType<T, ARank> &A) -> std::enable_if_t<sizeof...(CIndices) == sizeof...(AIndices) && sizeof...(CIndices) == CRank &&
                                                        sizeof...(AIndices) == ARank && std::is_arithmetic_v<U>> {

    LabeledSection1((std::fabs(UC_prefactor) > EINSUMS_ZERO)
                        ? fmt::format(R"(sort: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor,
                                      A.name(), print_tuple_no_type(A_indices), UC_prefactor, C->name(), print_tuple_no_type(C_indices))
                        : fmt::format(R"(sort: "{}"{} = {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor, A.name(),
                                      print_tuple_no_type(A_indices)));

    const T C_prefactor = UC_prefactor;
    const T A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a sort
    constexpr auto check = difference_t<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    // Librett uses the reverse order for indices.
    auto target_position_in_A = detail::find_type_with_position(detail::reverse_inds(C_indices), detail::reverse_inds(A_indices));

    // LibreTT interface currently only works for full Tensors and not TensorViews
#if defined(EINSUMS_USE_HPTT)
    if constexpr (std::is_same_v<CType<T, CRank>, DeviceTensor<T, CRank>> && std::is_same_v<AType<T, ARank>, DeviceTensor<T, ARank>>) {
        if (C_prefactor == T{0.0}) {
            std::array<int, ARank> perms{};
            std::array<int, ARank> size{};

            for (int i0 = 0; i0 < ARank; i0++) {
                perms[i0] = get_from_tuple<unsigned long>(target_position_in_A, (2 * i0) + 1);
                size[i0]  = A.dim(ARank - i0 - 1);
            }

            using T_devtype  = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<decltype(C->gpu_data())>>>;
            using T_hosttype = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<T>>>;

            detail::gpu_sort(perms.data(), ARank, einsums::gpu::HipCast<T_devtype, T_hosttype>::cast(A_prefactor), A.gpu_data(),
                             size.data(), einsums::gpu::HipCast<T_devtype, T_hosttype>::cast(C_prefactor), C->gpu_data());
            if (A_prefactor != T{1.0}) {
                *C *= A_prefactor; // Librett does not handle prefactors (yet?)
            }

            gpu::stream_wait();
            return;
        }
    }
#endif
    if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)>) {
        ::einsums::linear_algebra::axpby(A_prefactor, A, C_prefactor, C);
    } else {
        int *index_table = new int[sizeof...(AIndices)];
        int *gpu_index_table;

        gpu::hip_catch(hipMalloc((void **)&gpu_index_table, sizeof...(AIndices) * sizeof(int)));

        einsums::tensor_algebra::detail::compile_index_table(A_indices, C_indices, index_table);

        gpu::hip_catch(hipMemcpy((void *)gpu_index_table, (void *)index_table, sizeof...(AIndices) * sizeof(int), hipMemcpyHostToDevice));

        delete[] index_table; // core version no longer needed.

        size_t *stride_A_gpu, *stride_C_gpu;

        gpu::hip_catch(hipMalloc((void **)&stride_A_gpu, sizeof...(AIndices) * sizeof(size_t)));
        gpu::hip_catch(hipMalloc((void **)&stride_C_gpu, sizeof...(AIndices) * sizeof(size_t)));

        gpu::hip_catch(
            hipMemcpy((void *)stride_A_gpu, (void *)A.strides().data(), sizeof(size_t) * sizeof...(AIndices), hipMemcpyHostToDevice));
        gpu::hip_catch(
            hipMemcpy((void *)stride_C_gpu, (void *)C->strides().data(), sizeof(size_t) * sizeof...(AIndices), hipMemcpyHostToDevice));

        hipStream_t stream = gpu::get_stream();

        using T_devtype  = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<decltype(C->gpu_data())>>>;
        using T_hosttype = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<T>>>;

        detail::sort_kernel<T_devtype, ARank><<<gpu::blocks(A.size()), gpu::block_size(A.size()), 0, stream>>>(
            gpu_index_table, gpu::HipCast<T_devtype, T_hosttype>::cast(A_prefactor), A.gpu_data(), stride_A_gpu,
            gpu::HipCast<T_devtype, T_hosttype>::cast(C_prefactor), C->gpu_data(), stride_C_gpu, A.size());
        hipEvent_t wait_event;

        gpu::hip_catch(hipEventCreate(&wait_event));
        gpu::hip_catch(hipEventRecord(wait_event, stream));

        gpu::hip_catch(hipFreeAsync(gpu_index_table, stream));
        gpu::hip_catch(hipFreeAsync(stride_A_gpu, stream));
        gpu::hip_catch(hipFreeAsync(stride_C_gpu, stream));

        gpu::hip_catch(hipEventSynchronize(wait_event));

        gpu::hip_catch(hipEventDestroy(wait_event));
    }
}

} // namespace einsums::tensor_algebra