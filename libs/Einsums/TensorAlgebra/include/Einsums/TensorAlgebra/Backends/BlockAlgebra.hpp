//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#include <stdexcept>
#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUStreams/GPUStreams.hpp>
#endif

#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <tuple>
namespace einsums::tensor_algebra::detail {

template <bool OnlyUseGenericAlgorithm, BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(std::tuple_size_v<IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>> != 0)
auto einsum_special_dispatch(typename CType::ValueType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             std::conditional_t<(sizeof(typename AType::ValueType) > sizeof(typename BType::ValueType)),
                                                typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    // Check compatibility.
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != B.block_dim(i) || A.block_dim(i) != C->block_dim(i)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Block tensors need to have the same block sizes.");
        }
    }

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices, A[i], B_indices, B[i]);
    }
}

template <bool OnlyUseGenericAlgorithm, BlockTensorConcept AType, BlockTensorConcept BType, BasicTensorConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires requires {
        requires(std::tuple_size_v<IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>> != 0);
        requires CType::Rank >= 1;
    }
auto einsum_special_dispatch(typename CType::ValueType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             std::conditional_t<(sizeof(typename AType::ValueType) > sizeof(typename BType::ValueType)),
                                                typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    // Check compatibility.
    if (A.num_blocks() != B.num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != B.block_dim(i)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Block tensors need to have the same block sizes.");
        }
    }

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        std::array<Range, CType::Rank> view_index;
        view_index.fill(A.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(std::apply(*C, view_index)), AB_prefactor, A_indices, A[i], B_indices,
                                        B[i]);
    }
}

// template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
//           template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
//           typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
//     requires requires {
//         requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
//         requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
//         requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
//         requires(std::tuple_size_v<intersect_t<std::tuple<AIndices...>, std::tuple<BIndices...>>> == 0);
//         requires CRank >= 1;
//     }
// auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
//                              const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
//                              const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
//                              const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {
//     static_assert(false, "Needs to be implemented.");
// }

template <bool OnlyUseGenericAlgorithm, BlockTensorConcept AType, BlockTensorConcept BType, ScalarConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
auto einsum_special_dispatch(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             std::conditional_t<(sizeof(typename AType::ValueType) > sizeof(typename BType::ValueType)),
                                                typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    using CDataType = ValueTypeT<CType>;

    // Check compatibility.
    if (A.num_blocks() != B.num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != B.block_dim(i)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Block tensors need to have the same block sizes.");
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    if constexpr (IsDeviceTensorV<AType>) {

        size_t     elems = omp_get_max_threads();
        std::vector<CDataType> temp(elems);

        for (int i = 0; i < elems; i++) {
            temp[i] = CDataType{0.0};
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            CDataType tempC;
            einsum<OnlyUseGenericAlgorithm>(CDataType{0.0}, C_indices, &tempC, AB_prefactor, A_indices, A[i], B_indices, B[i]);
            temp[omp_get_thread_num()] += (CDataType)tempC;
        }

        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }

        *C = std::accumulate(temp.cbegin(), temp.cend(), (CDataType)*C);
    } else {
#endif
        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }
        CDataType temp = *C;

        // Loop through and perform einsums.
#pragma omp parallel for reduction(+ : temp)
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            CType temp_c = *C;
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, &temp_c, AB_prefactor, A_indices, A[i], B_indices, B[i]);
            temp += (CDataType)temp_c;
        }
        *C += temp;
#ifdef EINSUMS_COMPUTE_CODE
    }
#endif
}

template <bool OnlyUseGenericAlgorithm, BasicTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(VectorConcept<AType>)
auto einsum_special_dispatch(typename CType::ValueType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    // Check compatibility.
    if (B.num_blocks() != C->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < B.num_blocks(); i++) {
        if (B.block_dim(i) != C->block_dim(i)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Block tensors need to have the same block sizes.");
        }
    }

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < B.num_blocks(); i++) {
        if (B.block_dim(i) == 0) {
            continue;
        }
        std::array<Range, AType::Rank> view_index;
        view_index.fill(B.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices, std::apply(A, view_index),
                                        B_indices, B[i]);
    }
}

template <bool OnlyUseGenericAlgorithm, BasicTensorConcept AType, BlockTensorConcept BType, BasicTensorConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires requires {
        requires VectorConcept<AType>;
        requires CType::Rank > 0;
    }
auto einsum_special_dispatch(typename CType::ValueType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < B.num_blocks(); i++) {
        if (B.block_dim(i) == 0) {
            continue;
        }
        std::array<Range, CType::Rank> view_index_c;
        view_index_c.fill(B.block_range(i));
        std::array<Range, AType::Rank> view_index_a;
        view_index_a.fill(B.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(std::apply(*C, view_index_c)), AB_prefactor, A_indices,
                                        std::apply(A, view_index_a), B_indices, B[i]);
    }
}

template <bool OnlyUseGenericAlgorithm, BasicTensorConcept AType, BlockTensorConcept BType, ScalarConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(VectorConcept<AType>)
auto einsum_special_dispatch(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    using CDataType = ValueTypeT<CType>;

#ifdef EINSUMS_COMPUTE_CODE
    if constexpr (IsDeviceTensorV<AType>) {
        CDataType *temp;

        size_t elems = omp_get_max_threads();
        hip_catch(hipMalloc(temp, elems * sizeof(CDataType)));
        std::vector<CDataType> host_temp(elems);
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < B.num_blocks(); i++) {
            if (B.block_dim(i) == 0) {
                continue;
            }
            std::array<Range, AType::Rank> view_index;
            view_index.fill(B.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, temp + omp_get_thread_num(), AB_prefactor, A_indices,
                                            std::apply(A, view_index), B_indices, B[i]);
        }

        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }

        hip_catch(hipMemcpy(host_temp.data(), temp, elems * sizeof(CDataType), hipMemcpyDeviceToHost));
        *C = std::accumulate(host_temp.cbegin(), host_temp.cend(), (CDataType)*C);

        hip_catch(hipFree(temp));
    } else {
#endif
        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }
        CDataType temp = *C;

        // Loop through and perform einsums.
#pragma omp parallel for reduction(+ : temp)
        for (int i = 0; i < B.num_blocks(); i++) {
            if (B.block_dim(i) == 0) {
                continue;
            }
            CType                          temp_c = *C;
            std::array<Range, AType::Rank> view_index;
            view_index.fill(B.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, &temp_c, AB_prefactor, A_indices, std::apply(A, view_index),
                                            B_indices, B[i]);
            temp += (CDataType)temp_c;
        }
        *C += temp;
#ifdef EINSUMS_COMPUTE_CODE
    }
#endif
}

template <bool OnlyUseGenericAlgorithm, BlockTensorConcept AType, BasicTensorConcept BType, BlockTensorConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(VectorConcept<BType>)
auto einsum_special_dispatch(typename CType::ValueType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    // Check compatibility.
    if (A.num_blocks() != C->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != C->block_dim(i)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Block tensors need to have the same block sizes.");
        }
    }

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        std::array<Range, BType::Rank> view_index;
        view_index.fill(A.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices, A[i], B_indices,
                                        std::apply(B, view_index));
    }
}

template <bool OnlyUseGenericAlgorithm, BlockTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires requires {
        requires CType::Rank > 0;
        requires VectorConcept<BType>;
    }
auto einsum_special_dispatch(typename CType::ValueType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        std::array<Range, CType::Rank> view_index_c;
        view_index_c.fill(A.block_range(i));
        std::array<Range, BType::Rank> view_index_b;
        view_index_b.fill(A.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(std::apply(*C, view_index_c)), AB_prefactor, A_indices, A[i], B_indices,
                                        std::apply(B, view_index_b));
    }
}

template <bool OnlyUseGenericAlgorithm, BlockTensorConcept AType, BasicTensorConcept BType, ScalarConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(VectorConcept<BType>)
auto einsum_special_dispatch(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {
    using CDataType = ValueTypeT<CType>;

#ifdef EINSUMS_COMPUTE_CODE
    if constexpr (IsDeviceTensorV<AType>) {
        CDataType *temp;

        size_t elems = omp_get_max_threads();

        hip_catch(hipMalloc(temp, elems * sizeof(CDataType)));
        std::vector<CDataType> host_temp(elems);
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            std::array<Range, BType::Rank> view_index;
            view_index.fill(A.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, temp + omp_get_thread_num(), AB_prefactor, A_indices, A[i],
                                            B_indices, std::apply(B, view_index));
        }

        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }

        hip_catch(hipMemcpy(host_temp.data(), temp, elems * sizeof(CDataType), hipMemcpyDeviceToHost));
        *C = std::accumulate(host_temp.cbegin(), host_temp.cend(), (CDataType)*C);

        hip_catch(hipFree(temp));
    } else {
#endif
        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }
        CDataType temp = *C;

        // Loop through and perform einsums.
#pragma omp parallel for reduction(+ : temp)
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            CType                          temp_c = *C;
            std::array<Range, BType::Rank> view_index;
            view_index.fill(A.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, &temp_c, AB_prefactor, A_indices, A[i], B_indices,
                                            std::apply(B, view_index));
            temp += temp_c;
        }
        *C += temp;
#ifdef EINSUMS_COMPUTE_CODE
    }
#endif
}

} // namespace einsums::tensor_algebra::detail