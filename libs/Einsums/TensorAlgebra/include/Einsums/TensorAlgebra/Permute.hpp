//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/SmartPointer.hpp>
#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorBase/Common.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/TensorAlgebra/Backends/DevicePermute.hpp>
#endif

namespace einsums::tensor_algebra {

#if !defined(EINSUMS_WINDOWS)
namespace detail {

void EINSUMS_EXPORT permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, float const beta, float *B);
void EINSUMS_EXPORT permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, double const beta,
                         double *B);
void EINSUMS_EXPORT permute(int const *perm, int const dim, std::complex<float> const alpha, std::complex<float> const *A, int const *sizeA,
                         std::complex<float> const beta, std::complex<float> *B);
void EINSUMS_EXPORT permute(int const *perm, int const dim, std::complex<double> const alpha, std::complex<double> const *A, int const *sizeA,
                         std::complex<double> const beta, std::complex<double> *B);

} // namespace detail
#endif

//
// permute algorithm
//
template <CoreTensorConcept AType, CoreTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires std::is_arithmetic_v<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
          std::tuple<AIndices...> const &A_indices, AType const &A) {
    using T                = typename AType::ValueType;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t CRank = CType::Rank;

    LabeledSection1((std::fabs(UC_prefactor) > EINSUMS_ZERO)
                        ? fmt::format(R"(permute: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor,
                                      A.name(), print_tuple_no_type(A_indices), UC_prefactor, C->name(), print_tuple_no_type(C_indices))
                        : fmt::format(R"(permute: "{}"{} = {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor, A.name(),
                                      print_tuple_no_type(A_indices)));

    T const C_prefactor = UC_prefactor;
    T const A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a permute
    constexpr auto check = DifferenceT<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    auto target_dims = get_dim_ranges<CRank>(*C);
    auto a_dims      = detail::get_dim_ranges_for(A, target_position_in_A);

    // If the prefactor is zero, set the tensor to zero. This avoids NaNs.
    if (C_prefactor == T{0.0}) {
        *C = T{0.0};
    }

    // HPTT interface currently only works for full Tensors and not TensorViews
#if !defined(EINSUMS_WINDOWS)
    if constexpr (std::is_same_v<CType, Tensor<T, CRank>> && std::is_same_v<AType, Tensor<T, ARank>>) {
        std::array<int, ARank> perms{};
        std::array<int, ARank> size{};

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0] = arguments::get_from_tuple<unsigned long>(target_position_in_A, (2 * i0) + 1);
            size[i0]  = A.dim(i0);
        }

        detail::permute(perms.data(), ARank, A_prefactor, A.data(), size.data(), C_prefactor, C->data());
    } else
#endif
        if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)>) {
        linear_algebra::axpby(A_prefactor, A, C_prefactor, C);
    } else {
        auto view = std::apply(ranges::views::cartesian_product, target_dims);

        EINSUMS_OMP_PARALLEL_FOR
        for (auto it = view.begin(); it < view.end(); it++) {
            auto A_order = detail::construct_indices<AIndices...>(*it, target_position_in_A, *it, target_position_in_A);

            T &target_value = std::apply(*C, *it);
            T  A_value      = std::apply(A, A_order);

            target_value = C_prefactor * target_value + A_prefactor * A_value;
        }
    }
}

// Sort with default values, no smart pointers
template <NotASmartPointer ObjectA, NotASmartPointer ObjectC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, ObjectC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    permute(0, C_indices, C, 1, A_indices, A);
}

// Sort with default values, two smart pointers
template <SmartPointer SmartPointerA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    permute(0, C_indices, C->get(), 1, A_indices, *A);
}

// Sort with default values, one smart pointer (A)
template <SmartPointer SmartPointerA, NotASmartPointer PointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, PointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    permute(0, C_indices, C, 1, A_indices, *A);
}

// Sort with default values, one smart pointer (C)
template <NotASmartPointer ObjectA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    permute(0, C_indices, C->get(), 1, A_indices, A);
}

template <BlockTensorConcept AType, BlockTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires InSamePlace<AType, CType>;
        requires std::is_arithmetic_v<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
          std::tuple<AIndices...> const &A_indices, AType const &A) {

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        auto       &C_block = (*C)[i];
        auto const &A_block = A[i];
        permute(UC_prefactor, C_indices, &C_block, UA_prefactor, A_indices, A_block);
    }
}

/**
 * @brief Finds the tile grid dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <typename CType, TensorConcept AType, TensorConcept BType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename... AllUniqueIndices>
inline auto get_grid_ranges_for_many(CType const &C, ::std::tuple<CIndices...> const &C_indices, AType const &A,
                                     ::std::tuple<AIndices...> const         &A_indices,
                                     ::std::tuple<AllUniqueIndices...> const &All_unique_indices) {
    return ::std::array{get_grid_ranges_for_many_a<AllUniqueIndices, 0>(C, C_indices, A, A_indices)...};
}

template <TiledTensorConcept AType, TiledTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires InSamePlace<AType, CType>;
        requires std::is_arithmetic_v<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
          std::tuple<AIndices...> const &A_indices, AType const &A) {

    using ADataType        = typename AType::ValueType;
    using CDataType        = typename CType::ValueType;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t CRank = CType::Rank;

    constexpr auto unique_indices = UniqueT<std::tuple<CIndices..., AIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_grid, unique_strides);

    std::array<int, ARank> A_index_table;
    std::array<int, CRank> C_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < unique_grid[0] * unique_strides[0]; sentinel++) {
        thread_local std::array<size_t, std::tuple_size<decltype(unique_indices)>::value> unique_index_table;

        sentinel_to_indices(sentinel, unique_strides, unique_index_table);
        thread_local std::array<int, ARank> A_tile_index;
        thread_local std::array<int, CRank> C_tile_index;

        for (int i = 0; i < ARank; i++) {
            A_tile_index[i] = unique_index_table[A_index_table[i]];
        }

        for (int i = 0; i < CRank; i++) {
            C_tile_index[i] = unique_index_table[C_index_table[i]];
        }

        if (!A.has_tile(A_tile_index) || A.has_zero_size(A_tile_index) || C->has_zero_size(C_tile_index)) {
            continue;
        }

        C->lock();
        auto &C_tile = C->tile(C_tile_index);
        C->unlock();
        C_tile.lock();
        permute(UC_prefactor, C_indices, &C_tile, UA_prefactor, A_indices, A.tile(A_tile_index));
        C_tile.unlock();
    }
}

template <typename... Args>
[[deprecated("The name sort is confusing, since we are not sorting values. Please use permute instead. This function will be removed in the future.")]]
inline auto sort(Args &&...args) -> decltype(permute(std ::forward<Args>(args)...)) {
    return permute(std ::forward<Args>(args)...);
}

} // namespace einsums::tensor_algebra