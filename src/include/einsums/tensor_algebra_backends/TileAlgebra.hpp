#pragma once

#include "einsums/TensorAlgebra.hpp"
#include "einsums/tensor_algebra_backends/Dispatch.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <tuple>

namespace einsums::tensor_algebra::detail {

template <typename UniqueIndex, int BDim, TensorConcept BType>
inline size_t get_grid_ranges_for_many_b(const BType &B, const ::std::tuple<> &B_indices) {
    return 1;
}

template <typename UniqueIndex, int BDim, TensorConcept BType, typename BHead>
inline auto get_grid_ranges_for_many_b(const BType &B, const ::std::tuple<BHead> &B_indices)
    -> ::std::enable_if<::std::is_same_v<BHead, UniqueIndex>, size_t> {
    if constexpr (einsums::detail::IsTiledTensorV<BType>) {
        return B.grid_size(BDim);
    } else if constexpr (einsums::detail::IsBlockTensorV<BType>) {
        return B.num_blocks();
    } else {
        return 1;
    }
}

template <typename UniqueIndex, int BDim, TensorConcept BType, typename BHead,
          typename... BIndices>
inline size_t get_grid_ranges_for_many_b(const BType &B, const ::std::tuple<BHead, BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<BHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<BType>) {
            return B.grid_size(BDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<BType>) {
            return B.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, BDim + 1>(B, ::std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, TensorConcept AType, 
          TensorConcept BType, typename... BIndices>
inline size_t get_grid_ranges_for_many_a(const AType &A, const ::std::tuple<> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, TensorConcept AType,
          TensorConcept BType, typename AHead, typename... BIndices>
inline size_t get_grid_ranges_for_many_a(const AType &A, const ::std::tuple<AHead> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<AType>) {
            return A.grid_size(ADim);
        } else if constexpr (einsums::detail::IsBlockTensorV<AType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, TensorConcept AType,
          TensorConcept BType, typename AHead, typename... AIndices,
          typename... BIndices>
inline auto get_grid_ranges_for_many_a(const AType&A, const ::std::tuple<AHead, AIndices...> &A_indices,
                                       const BType   &B,
                                       const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(AIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<AType>) {
            return A.grid_size(ADim);
        } else if constexpr (einsums::detail::IsBlockTensorV<AType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, ADim + 1>(A, ::std::tuple<AIndices...>(), B, B_indices);
    }
}

// In these functions, leave CType as typename to allow for scalar types and tensor types.
template <typename UniqueIndex, int CDim, typename CType,
          TensorConcept AType, TensorConcept BType,
          typename... AIndices, typename... BIndices>
inline size_t get_grid_ranges_for_many_c(const CType &C, const ::std::tuple<> &C_indices,
                                         const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim,  typename CType,
          TensorConcept AType, TensorConcept BType,
          typename CHead, typename... AIndices, typename... BIndices>
inline size_t get_grid_ranges_for_many_c(const CType &C, const ::std::tuple<CHead> &C_indices,
                                         const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                         const BType &B, const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<CType>) {
            return C.grid_size(CDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<CType>) {
            return C.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim,  typename CType,
          TensorConcept AType, TensorConcept BType,
          typename CHead, typename... CIndices, typename... AIndices, typename... BIndices>
inline auto get_grid_ranges_for_many_c(const CType &C, const ::std::tuple<CHead, CIndices...> &C_indices,
                                       const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                       const BType   &B,
                                       const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(CIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (einsums::detail::IsTiledTensorV<CType>) {
            return C.grid_size(CDim);
        } else if constexpr (einsums::detail::IsBlockTensorV<CType>) {
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
template <typename CType, TensorConcept AType,
          TensorConcept BType,
          typename... CIndices, typename... AIndices, typename... BIndices, typename... AllUniqueIndices>
inline auto get_grid_ranges_for_many(const CType &C, const ::std::tuple<CIndices...> &C_indices,
                                     const AType &A, const ::std::tuple<AIndices...> &A_indices,
                                     const BType &B, const ::std::tuple<BIndices...> &B_indices,
                                     const ::std::tuple<AllUniqueIndices...> &All_unique_indices) {
    return ::std::array{get_grid_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

template<bool OnlyUseGenericAlgorithm, TiledTensorConcept AType, TiledTensorConcept BType, TiledTensorConcept CType, typename... CIndices, typename... AIndices, typename... BIndices>
requires(CType::rank != 0)
auto einsum_special_dispatch(const typename CType::data_type C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C,
                             const BiggestTypeT<typename AType::data_type, typename BType::data_type> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType &A,
                             const std::tuple<BIndices...> &B_indices, const BType &B) -> void {

    constexpr size_t ARank = AType::rank;
    constexpr size_t BRank = BType::rank;
    constexpr size_t CRank = CType::rank;

    using ADataType = typename AType::data_type;
    using BDataType = typename BType::data_type;
    using CDataType = typename CType::data_type;

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

template<bool OnlyUseGenericAlgorithm, TiledTensorConcept AType, TiledTensorConcept BType, ScalarConcept CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum_special_dispatch(const DataTypeT<CType> C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C,
                             const BiggestTypeT<typename AType::data_type, typename BType::data_type> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType &A,
                             const std::tuple<BIndices...> &B_indices, const BType &B) -> void {

    constexpr size_t ARank = AType::rank;
    constexpr size_t BRank = BType::rank;
    constexpr size_t CRank = 0;

    using ADataType = typename AType::data_type;
    using BDataType = typename BType::data_type;
    using CDataType = typename CType::data_type;

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

#pragma omp parallel for reduction(+ : out)
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

            CDataType C_tile;
            einsum<OnlyUseGenericAlgorithm>(CDataType{0.0}, C_indices, &C_tile, AB_prefactor, A_indices, A.tile(A_tile_index), B_indices,
                                            B.tile(B_tile_index));
            out += C_tile;
    }

    *C += out;
}
} // namespace einsums::tensor_algebra::detail