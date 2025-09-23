//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TypeSupport/GPUCast.hpp>

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace einsums::tensor_algebra {

namespace detail {

void EINSUMS_EXPORT gpu_permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, float const beta,
                                float *B);
void EINSUMS_EXPORT gpu_permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, double const beta,
                                double *B);
void EINSUMS_EXPORT gpu_permute(int const *perm, int const dim, hipFloatComplex const alpha, hipFloatComplex const *A, int const *sizeA,
                                hipFloatComplex const beta, hipFloatComplex *B);
void EINSUMS_EXPORT gpu_permute(int const *perm, int const dim, hipDoubleComplex const alpha, hipDoubleComplex const *A, int const *sizeA,
                                hipDoubleComplex const beta, hipDoubleComplex *B);

template <bool ConjA, typename T, size_t Rank>
__global__ void permute_kernel(int const *perm, T const alpha, T const *A, size_t const *strideA, T const beta, T *B, size_t const *strideB,
                               size_t size) {
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t A_index[Rank], B_index[Rank];
    size_t A_sentinel, B_sentinel;

    for (ptrdiff_t curr_index = thread_id; curr_index < size; curr_index++) {
        sentinel_to_indices<Rank>(curr_index, strideA, A_index);
        A_sentinel = 0;
        B_sentinel = 0;

#pragma unroll
        for (ptrdiff_t i = 0; i < Rank; i++) {
            A_sentinel += strideA[i] * A_index[i];
        }

#pragma unroll
        for (ptrdiff_t i = 0; i < Rank; i++) {
            B_sentinel += strideB[i] * A_index[perm[i]];
        }

        if constexpr (ConjA && IsComplexV<T>) {
            B[B_sentinel] = gpu_ops::fma(alpha, gpu_ops::conj(A[A_sentinel]), beta * B[B_sentinel]);
        } else {
            B[B_sentinel] = gpu_ops::fma(alpha, A[A_sentinel], beta * B[B_sentinel]);
        }
    }
}

template <typename Index>
auto reverse_inds(std::tuple<Index> const &ind) {
    return ind;
}

template <typename Head, typename... Tail>
    requires(sizeof...(Tail) > 0)
auto reverse_inds(std::tuple<Head, Tail...> const &inds) {
    return std::tuple_cat(reverse_inds(std::tuple<Tail...>{}), std::tuple<Head>{});
}

} // namespace detail

//
// permute algorithm
//
template <bool ConjA = false, template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename CType,
          size_t CRank, typename... CIndices, typename... AIndices, typename U, typename T = double>
    requires requires {
        requires DeviceRankTensor<AType<T, ARank>, ARank, T>;
        requires DeviceRankTensor<CType<T, CRank>, CRank, T>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
auto permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType<T, CRank> *C, U const UA_prefactor,
             std::tuple<AIndices...> const &A_indices, AType<T, ARank> const &A)
    -> std::enable_if_t<sizeof...(CIndices) == sizeof...(AIndices) && sizeof...(CIndices) == CRank && sizeof...(AIndices) == ARank &&
                        std::is_arithmetic_v<U>> {

    LabeledSection(
        fmt::runtime((std::abs(UC_prefactor) > EINSUMS_ZERO)
                         ? fmt::format(R"(permute: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), C_indices, UA_prefactor, A.name(),
                                       A_indices, UC_prefactor, C->name(), C_indices)
                         : fmt::format(R"(permute: "{}"{} = {} "{}"{})", C->name(), C_indices, UA_prefactor, A.name(), A_indices)));

    T const C_prefactor = UC_prefactor;
    T const A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a permute
    constexpr auto check = DifferenceT<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    // Librett uses the reverse order for indices.
    auto target_position_in_A = detail::find_type_with_position(detail::reverse_inds(C_indices), detail::reverse_inds(A_indices));

    auto check_target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    einsums::for_sequence<ARank>([&](auto n) {
        if (C->dim((size_t)n) < A.dim(std::get<2 * (size_t)n + 1>(check_target_position_in_A))) {
            EINSUMS_THROW_EXCEPTION(dimension_error,
                                    "The {} dimension of the output tensor is smaller than the {} dimension of the input tensor!",
                                    print::ordinal((size_t)n), print::ordinal(std::get<2 * (size_t)n + 1>(check_target_position_in_A)));
        }

        if (C->dim((size_t)n) == 0) {
            return;
        }
    });

    // LibreTT interface currently only works for full Tensors and not TensorViews
#if defined(EINSUMS_USE_LIBRETT)
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

            detail::gpu_permute(perms.data(), ARank, HipCast<T_devtype, T_hosttype>::cast(A_prefactor), A.gpu_data(), size.data(),
                                HipCast<T_devtype, T_hosttype>::cast(C_prefactor), C->gpu_data());
            if (A_prefactor != T{1.0}) {
                *C *= A_prefactor; // Librett does not handle prefactors (yet?)
            }

            gpu::stream_wait();
            return;
        }
    }
#endif
    if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)> && !(ConjA && IsComplexV<T>)) {
        einsums::linear_algebra::axpby(A_prefactor, A, C_prefactor, C);
    } else {
        int *index_table = new int[sizeof...(AIndices)];
        int *gpu_index_table;

        hip_catch(hipMalloc((void **)&gpu_index_table, sizeof...(AIndices) * sizeof(int)));

        compile_index_table(A_indices, C_indices, index_table);

        hip_catch(hipMemcpy((void *)gpu_index_table, (void *)index_table, sizeof...(AIndices) * sizeof(int), hipMemcpyHostToDevice));

        delete[] index_table; // core version no longer needed.

        size_t *stride_A_gpu, *stride_C_gpu;

        hip_catch(hipMalloc((void **)&stride_A_gpu, sizeof...(AIndices) * sizeof(size_t)));
        hip_catch(hipMalloc((void **)&stride_C_gpu, sizeof...(AIndices) * sizeof(size_t)));

        hip_catch(hipMemcpy((void *)stride_A_gpu, (void *)A.strides().data(), sizeof(size_t) * sizeof...(AIndices), hipMemcpyHostToDevice));
        hip_catch(
            hipMemcpy((void *)stride_C_gpu, (void *)C->strides().data(), sizeof(size_t) * sizeof...(AIndices), hipMemcpyHostToDevice));

        hipStream_t stream = gpu::get_stream();

        using T_devtype  = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<decltype(C->gpu_data())>>>;
        using T_hosttype = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<T>>>;

        detail::permute_kernel<ConjA, T_devtype, ARank><<<gpu::blocks(A.size()), gpu::block_size(A.size()), 0, stream>>>(
            gpu_index_table, HipCast<T_devtype, T_hosttype>::cast(A_prefactor), A.gpu_data(), stride_A_gpu,
            HipCast<T_devtype, T_hosttype>::cast(C_prefactor), C->gpu_data(), stride_C_gpu, A.size());
        hipEvent_t wait_event;

        hip_catch(hipEventCreate(&wait_event));
        hip_catch(hipEventRecord(wait_event, stream));

        hip_catch(hipFreeAsync(gpu_index_table, stream));
        hip_catch(hipFreeAsync(stride_A_gpu, stream));
        hip_catch(hipFreeAsync(stride_C_gpu, stream));

        hip_catch(hipEventSynchronize(wait_event));

        hip_catch(hipEventDestroy(wait_event));
    }
}

} // namespace einsums::tensor_algebra