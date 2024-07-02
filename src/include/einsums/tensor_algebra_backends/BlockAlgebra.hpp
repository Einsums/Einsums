#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include <stdexcept>
#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

#include "einsums/utility/TensorTraits.hpp"

#include <tuple>
namespace einsums::tensor_algebra::detail {

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBlockTensor<CType<CDataType, CRank>, CRank, CDataType>;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    // Check compatibility.
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks()) {
        throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != B.block_dim(i) || A.block_dim(i) != C->block_dim(i)) {
            throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same block sizes.");
        }
    }

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices, A[i], B_indices, B[i]);
    }
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires CRank >= 1;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    // Check compatibility.
    if (A.num_blocks() != B.num_blocks()) {
        throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != B.block_dim(i)) {
            throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same block sizes.");
        }
    }


    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        std::array<Range, CRank> view_index;
        view_index.fill(A.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(std::apply(*C, view_index)), AB_prefactor, A_indices, A[i], B_indices,
                                        B[i]);
    }
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires CRank == 0;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    // Check compatibility.
    if (A.num_blocks() != B.num_blocks()) {
        throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != B.block_dim(i)) {
            throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same block sizes.");
        }
    }

#ifdef __HIP__
    if constexpr (einsums::detail::IsDeviceRankTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
        __device_ptr__ CDataType *temp;

        gpu::hip_catch(hipMalloc(temp, omp_get_max_threads() * sizeof(CDataType)));
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, temp + omp_get_thread_num(), AB_prefactor, A_indices, A[i], B_indices,
                                            B[i]);
        }

        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }
        __host_ptr__ CDataType *host_temp = new CDataType[omp_get_max_threads()];

        gpu::hip_catch(hipMemcpy(host_temp, temp, omp_get_max_threads() * sizeof(CDataType), hipMemcpyDeviceToHost));
        *C = std::accumulate(host_temp, host_temp + omp_get_max_threads(), (CDataType)*C);

        delete[] host_temp;
        gpu::hip_catch(hipFree(temp));
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
            CType<CDataType, CRank> temp_c = *C;
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, &temp_c, AB_prefactor, A_indices, A[i], B_indices, B[i]);
            temp += (CDataType)temp_c;
        }
        *C += temp;
#ifdef __HIP__
    }
#endif
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBasicTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBlockTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires ARank == 1;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    // Check compatibility.
    if (B.num_blocks() != C->num_blocks()) {
        throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < B.num_blocks(); i++) {
        if (B.block_dim(i) != C->block_dim(i)) {
            throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same block sizes.");
        }
    }

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < B.num_blocks(); i++) {
        std::array<Range, ARank> view_index;
        view_index.fill(B.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices, std::apply(A, view_index),
                                        B_indices, B[i]);
    }
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBasicTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires CRank >= 1;
        requires ARank == 1;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < B.num_blocks(); i++) {
        std::array<Range, CRank> view_index_c;
        view_index_c.fill(B.block_range(i));
        std::array<Range, ARank> view_index_a;
        view_index_a.fill(B.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(std::apply(*C, view_index_c)), AB_prefactor, A_indices,
                                        std::apply(A, view_index_a), B_indices, B[i]);
    }
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBasicTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires CRank == 0;
        requires ARank == 1;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

#ifdef __HIP__
    if constexpr (einsums::detail::IsDeviceRankTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
        __device_ptr__ CDataType *temp;

        gpu::hip_catch(hipMalloc(temp, omp_get_max_threads() * sizeof(CDataType)));
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < B.num_blocks(); i++) {
            std::array<Range, ARank> view_index;
            view_index.fill(B.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, temp + omp_get_thread_num(), AB_prefactor, A_indices,
                                            std::apply(A, view_index), B_indices, B[i]);
        }

        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }
        __host_ptr__ CDataType *host_temp = new CDataType[omp_get_max_threads()];

        gpu::hip_catch(hipMemcpy(host_temp, temp, omp_get_max_threads() * sizeof(CDataType), hipMemcpyDeviceToHost));
        *C = std::accumulate(host_temp, host_temp + omp_get_max_threads(), (CDataType)*C);

        delete[] host_temp;
        gpu::hip_catch(hipFree(temp));
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
            CType<CDataType, CRank>  temp_c = *C;
            std::array<Range, ARank> view_index;
            view_index.fill(B.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, &temp_c, AB_prefactor, A_indices, std::apply(A, view_index), B_indices,
                                            B[i]);
            temp += (CDataType)temp_c;
        }
        *C += temp;
#ifdef __HIP__
    }
#endif
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBasicTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBlockTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires BRank == 1;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    // Check compatibility.
    if (A.num_blocks() != C->num_blocks()) {
        throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same number of blocks.");
    }

    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != C->block_dim(i)) {
            throw std::runtime_error("einsum_special_dispatch: Block tensors need to have the same block sizes.");
        }
    }

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        std::array<Range, BRank> view_index;
        view_index.fill(A.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices, A[i], B_indices,
                                        std::apply(B, view_index));
    }
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBasicTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires CRank >= 1;
        requires BRank == 1;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    // Loop through and perform einsums.
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        std::array<Range, CRank> view_index_c;
        view_index_c.fill(A.block_range(i));
        std::array<Range, BRank> view_index_b;
        view_index_b.fill(A.block_range(i));
        einsum<OnlyUseGenericAlgorithm>(C_prefactor, C_indices, &(std::apply(*C, view_index_c)), AB_prefactor, A_indices, A[i], B_indices,
                                        std::apply(B, view_index_b));
    }
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBasicTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBasicTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires CRank == 0;
        requires BRank == 1;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

#ifdef __HIP__
    if constexpr (einsums::detail::IsDeviceRankTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
        __device_ptr__ CDataType *temp;

        gpu::hip_catch(hipMalloc(temp, omp_get_max_threads() * sizeof(CDataType)));
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            std::array<Range, BRank> view_index;
            view_index.fill(A.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, temp + omp_get_thread_num(), AB_prefactor, A_indices, A[i], B_indices,
                                            std::apply(B, view_index));
        }

        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }
        __host_ptr__ CDataType *host_temp = new CDataType[omp_get_max_threads()];

        gpu::hip_catch(hipMemcpy(host_temp, temp, omp_get_max_threads() * sizeof(CDataType), hipMemcpyDeviceToHost));
        *C = std::accumulate(host_temp, host_temp + omp_get_max_threads(), (CDataType)*C);

        delete[] host_temp;
        gpu::hip_catch(hipFree(temp));
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
            CType<CDataType, CRank>  temp_c = *C;
            std::array<Range, BRank> view_index;
            view_index.fill(A.block_range(i));
            einsum<OnlyUseGenericAlgorithm>(CDataType(0.0), C_indices, &temp_c, AB_prefactor, A_indices, A[i], B_indices,
                                            std::apply(B, view_index));
            temp += (CDataType)temp_c;
        }
        *C += temp;
#ifdef __HIP__
    }
#endif
}

} // namespace einsums::tensor_algebra::detail