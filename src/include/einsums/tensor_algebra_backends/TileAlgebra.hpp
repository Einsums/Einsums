#pragma once

#include "einsums/TensorAlgebra.hpp"
#include "einsums/tensor_algebra_backends/Dispatch.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <tuple>

namespace einsums::tensor_algebra::detail {

template <typename UniqueIndex, int BDim, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
inline size_t get_grid_ranges_for_many_b(const BType<BDataType, BRank> &B, const ::std::tuple<> &B_indices) {
    return 1;
}

template <typename UniqueIndex, int BDim, template <typename, size_t> typename BType, typename BDataType, size_t BRank, typename BHead>
inline auto get_grid_ranges_for_many_b(const BType<BDataType, BRank> &B, const ::std::tuple<BHead> &B_indices)
    -> ::std::enable_if<::std::is_same_v<BHead, UniqueIndex>, size_t> {
    if constexpr (einsums::detail::IsTiledTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        return B.grid_size(BDim);
    } else if constexpr (einsums::detail::IsBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        return B.num_blocks();
    } else {
        return 1;
    }
}

template <typename UniqueIndex, int BDim, template <typename, size_t> typename BType, typename BDataType, size_t BRank, typename BHead,
          typename... BIndices>
inline size_t get_grid_ranges_for_many_b(const BType<BDataType, BRank> &B, const ::std::tuple<BHead, BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<BHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
            return B.grid_size(BDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
            return B.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, BDim + 1>(B, ::std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, typename... BIndices>
inline size_t get_grid_ranges_for_many_a(const AType<ADataType, ARank> &A, const ::std::tuple<> &A_indices,
                                         const BType<BDataType, BRank> &B, const ::std::tuple<BIndices...> &B_indices) {
    return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, typename AHead, typename... BIndices>
inline size_t get_grid_ranges_for_many_a(const AType<ADataType, ARank> &A, const ::std::tuple<AHead> &A_indices,
                                         const BType<BDataType, BRank> &B, const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
            return A.grid_size(ADim);
        } else if constexpr (einsums::detail::IsBlockTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, typename AHead, typename... AIndices,
          typename... BIndices>
inline auto get_grid_ranges_for_many_a(const AType<ADataType, ARank> &A, const ::std::tuple<AHead, AIndices...> &A_indices,
                                       const BType<BDataType, BRank>   &B,
                                       const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(AIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
            return A.grid_size(ADim);
        } else if constexpr (einsums::detail::IsBlockTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, ADim + 1>(A, ::std::tuple<AIndices...>(), B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, typename... AIndices, typename... BIndices>
inline size_t get_grid_ranges_for_many_c(const CType<CDataType, CRank> &C, const ::std::tuple<> &C_indices,
                                         const AType<ADataType, ARank> &A, const ::std::tuple<AIndices...> &A_indices,
                                         const BType<BDataType, BRank> &B, const ::std::tuple<BIndices...> &B_indices) {
    return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, typename CHead, typename... AIndices, typename... BIndices>
inline size_t get_grid_ranges_for_many_c(const CType<CDataType, CRank> &C, const ::std::tuple<CHead> &C_indices,
                                         const AType<ADataType, ARank> &A, const ::std::tuple<AIndices...> &A_indices,
                                         const BType<BDataType, BRank> &B, const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
            return C.grid_size(CDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
            return C.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, typename CHead, typename... CIndices, typename... AIndices, typename... BIndices>
inline auto get_grid_ranges_for_many_c(const CType<CDataType, CRank> &C, const ::std::tuple<CHead, CIndices...> &C_indices,
                                       const AType<ADataType, ARank> &A, const ::std::tuple<AIndices...> &A_indices,
                                       const BType<BDataType, BRank>   &B,
                                       const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(CIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
            return C.grid_size(CDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
            return C.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_c<UniqueIndex, CDim + 1>(C, ::std::tuple<CIndices...>(), A, A_indices, B, B_indices);
    }
}

/**
 * @brief Finds the tile grid dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param B The B tensor.
 * @param B_indices The indices for the B tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <template <typename, size_t> typename CType, typename CDataType, size_t CRank, template <typename, size_t> typename AType,
          typename ADataType, size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... AllUniqueIndices>
inline auto get_grid_ranges_for_many(const CType<CDataType, CRank> &C, const ::std::tuple<CIndices...> &C_indices,
                                     const AType<ADataType, ARank> &A, const ::std::tuple<AIndices...> &A_indices,
                                     const BType<BDataType, BRank> &B, const ::std::tuple<BIndices...> &B_indices,
                                     const ::std::tuple<AllUniqueIndices...> &All_unique_indices) {
    return ::std::array{get_grid_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankTiledTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankTiledTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankTiledTensor<CType<CDataType, CRank>, CRank, CDataType>;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    constexpr auto unique_indices = unique_t<std::tuple<CIndices..., AIndices..., BIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_grid, unique_strides);

    std::array<int, ARank> A_index_table;
    std::array<int, BRank> B_index_table;
    std::array<int, CRank> C_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    if (C_prefactor == CDataType(0.0)) {
        C->zero();
    } else {
        *C *= C_prefactor;
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < unique_grid[0] * unique_strides[0]; sentinel++) {
        thread_local std::array<size_t, std::tuple_size<decltype(unique_indices)>::value> unique_index_table;

        sentinel_to_indices(sentinel, unique_strides, unique_index_table);
        thread_local std::array<int, ARank> A_tile_index;
        thread_local std::array<int, BRank> B_tile_index;
        thread_local std::array<int, CRank> C_tile_index;

        for (int i = 0; i < ARank; i++) {
            A_tile_index[i] = unique_index_table[A_index_table[i]];
        }

        for (int i = 0; i < BRank; i++) {
            B_tile_index[i] = unique_index_table[B_index_table[i]];
        }

        for (int i = 0; i < CRank; i++) {
            C_tile_index[i] = unique_index_table[C_index_table[i]];
        }

        if (!A.has_tile(A_tile_index) || !B.has_tile(B_tile_index) || A.has_zero_size(A_tile_index) || B.has_zero_size(B_tile_index) ||
            C->has_zero_size(C_tile_index)) {
            continue;
        }

        C->lock();
        auto &C_tile = C->tile(C_tile_index);
        C->unlock();
        C_tile.lock();
        einsum<OnlyUseGenericAlgorithm>(CDataType{1.0}, C_indices, &C_tile, AB_prefactor, A_indices, A.tile(A_tile_index), B_indices,
                                        B.tile(B_tile_index));
        C_tile.unlock();
    }
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankTiledTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankTiledTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBasicTensor<CType<CDataType, CRank>, 0, CDataType>;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {

    constexpr auto unique_indices = unique_t<std::tuple<CIndices..., AIndices..., BIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_grid, unique_strides);

    std::array<int, ARank> A_index_table;
    std::array<int, BRank> B_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);

    if (C_prefactor == CDataType(0.0)) {
        *C = CDataType{0.0};
    } else {
        *C *= C_prefactor;
    }

    CDataType out{0.0};

#pragma omp parallel for simd reduction(+ : out)
    for (size_t sentinel = 0; sentinel < unique_grid[0] * unique_strides[0]; sentinel++) {
        thread_local std::array<size_t, std::tuple_size<decltype(unique_indices)>::value> unique_index_table;

        sentinel_to_indices(sentinel, unique_strides, unique_index_table);
        thread_local std::array<int, ARank> A_tile_index;
        thread_local std::array<int, BRank> B_tile_index;

        for (int i = 0; i < ARank; i++) {
            A_tile_index[i] = unique_index_table[A_index_table[i]];
        }

        for (int i = 0; i < BRank; i++) {
            B_tile_index[i] = unique_index_table[B_index_table[i]];
        }

        if(!A.has_tile(A_tile_index) || !B.has_tile(B_tile_index) || A.has_zero_size(A_tile_index) || B.has_zero_size(B_tile_index)) {
            continue;
        }

#ifdef __HIP__
        if constexpr (einsums::detail::IsDeviceRankTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
            DeviceTensor<CDataType, 0> C_tile;
            einsum<OnlyUseGenericAlgorithm>(CDataType{0.0}, C_indices, &C_tile, AB_prefactor, A_indices, A.tile(A_tile_index), B_indices,
                                            B.tile(B_tile_index));
            out += (double)C_tile;
        } else {
#endif
            Tensor<CDataType, 0> C_tile;
            einsum<OnlyUseGenericAlgorithm>(CDataType{0.0}, C_indices, &C_tile, AB_prefactor, A_indices, A.tile(A_tile_index), B_indices,
                                            B.tile(B_tile_index));
            out += (double)C_tile;
#ifdef __HIP__
        }
#endif
    }

    *C = out;
}
} // namespace einsums::tensor_algebra::detail