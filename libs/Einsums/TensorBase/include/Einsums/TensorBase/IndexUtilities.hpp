//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <range/v3/range_fwd.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>

#include <cstdarg>
#include <cstddef>
#include <source_location>
#include <tuple>
#include <type_traits>
#include <vector>

namespace einsums {

#ifndef DOXYGEN
template <typename UniqueIndex, int BDim, typename BType>
inline size_t get_dim_ranges_for_many_b(BType const &B, std::tuple<> const &B_indices) {
    return 1;
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead>
inline auto get_dim_ranges_for_many_b(BType const &B, std::tuple<BHead> const &B_indices)
    -> std::enable_if<std::is_same_v<BHead, UniqueIndex>, size_t> {
    return B.dim(BDim);
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_b(BType const &B, std::tuple<BHead, BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<BHead, UniqueIndex>) {
        return B.dim(BDim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, BDim + 1>(B, std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(AType const &A, std::tuple<> const &A_indices, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(AType const &A, std::tuple<AHead> const &A_indices, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_a(AType const &A, std::tuple<AHead, AIndices...> const &A_indices, BType const &B,
                                      std::tuple<BIndices...> const &B_indices) -> std::enable_if_t<sizeof...(AIndices) != 0, size_t> {
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, ADim + 1>(A, std::tuple<AIndices...>(), B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename... AIndices, typename... BIndices>
inline size_t get_dim_ranges_for_many_c(CType const &C, std::tuple<> const &C_indices, AType const &A,
                                        std::tuple<AIndices...> const &A_indices, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... AIndices,
          typename... BIndices>
inline size_t get_dim_ranges_for_many_c(CType const &C, std::tuple<CHead> const &C_indices, AType const &A,
                                        std::tuple<AIndices...> const &A_indices, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<CHead, UniqueIndex>) {
        return C.dim(CDim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... CIndices,
          typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_c(CType const &C, std::tuple<CHead, CIndices...> const &C_indices, AType const &A,
                                      std::tuple<AIndices...> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices)
    -> std::enable_if_t<sizeof...(CIndices) != 0, size_t> {
    if constexpr (std::is_same_v<CHead, UniqueIndex>) {
        return C.dim(CDim);
    } else {
        return get_dim_ranges_for_many_c<UniqueIndex, CDim + 1>(C, std::tuple<CIndices...>(), A, A_indices, B, B_indices);
    }
}

#endif

/**
 * @brief Finds the dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param B The B tensor.
 * @param B_indices The indices for the B tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <typename CType, typename AType, typename BType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename... AllUniqueIndices>
inline auto get_dim_ranges_for_many(CType const &C, std::tuple<CIndices...> const &C_indices, AType const &A,
                                    std::tuple<AIndices...> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices,
                                    std::tuple<AllUniqueIndices...> const &All_unique_indices) {
    return std::array{get_dim_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

namespace detail {

/**
 * @brief Get the dim ranges object
 *
 * @tparam TensorType
 * @tparam I
 * @param tensor The tensor object to query
 * @return A tuple containing the dimension ranges compatible with range-v3 cartesian_product function.
 */
template <typename TensorType, std::size_t... I>
auto get_dim_ranges(TensorType const &tensor, std::index_sequence<I...>) {
    return std::tuple{ranges::views::ints(0, (int)tensor.dim(I))...};
}

/**
 * @brief Adds elements from two sources into the target.
 *
 * Useful in adding offsets to a set of indices.
 *
 * @tparam N The rank of the data.
 * @tparam Target
 * @tparam Source1
 * @tparam Source2
 * @param target The output
 * @param source1 The first source
 * @param source2 The second source
 */
template <size_t N, typename Target, typename Source1, typename Source2>
void add_elements(Target &target, Source1 const &source1, Source2 const &source2) {
    if constexpr (N > 1) {
        add_elements<N - 1>(target, source1, source2);
    }
    target[N - 1] = source1[N - 1] + std::get<N - 1>(source2);
}

} // namespace detail

/**
 * @brief Find the ranges for each dimension of a tensor.
 *
 * The returned tuple is compatible with ranges-v3 cartesian_product function.
 *
 * @tparam N
 * @tparam TensorType
 * @tparam Rank
 * @tparam T
 * @param tensor Tensor to query
 * @return Tuple containing the range for each dimension of the tensor.
 */
template <int N, typename TensorType>
auto get_dim_ranges(TensorType const &tensor) {
    return detail::get_dim_ranges(tensor, std::make_index_sequence<N>{});
}

/**
 * @brief Converts a single sentinel value into a list of indices.
 *
 * @param sentinel The sentinel to convert.
 * @param unique_strides The strides of the unique indices that the sentinel is iterating over.
 * @param out_inds The converted indices.
 */
template <size_t num_unique_inds>
EINSUMS_HOSTDEV inline void sentinel_to_indices(size_t sentinel, size_t const *unique_strides, size_t *out_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ptrdiff_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            [[unlikely]] out_inds[i] = 0;
        }
    }
}

template <size_t num_unique_inds>
void sentinel_to_indices(size_t sentinel, std::array<size_t, num_unique_inds> const &unique_strides,
                         std::array<size_t, num_unique_inds> &out_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ptrdiff_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            [[unlikely]] out_inds[i] = 0;
        }
    }
}

template <typename StorageType1, typename StorageType2>
void sentinel_to_indices(size_t sentinel, StorageType1 const &unique_strides, StorageType2 &out_inds) {
    size_t hold = sentinel;

    // if (out_inds.size() != unique_strides.size()) {
    //     out_inds.resize(unique_strides.size());
    // }

    for (ptrdiff_t i = 0; i < unique_strides.size(); i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            [[unlikely]] out_inds[i] = 0;
        }
    }
}

EINSUMS_HOSTDEV inline void sentinel_to_indices_mult_imp(size_t) {
}

template <typename Extra>
EINSUMS_HOSTDEV void sentinel_to_indices_mult_imp(size_t ordinal, Extra extra) = delete;

template <typename... Rest>
EINSUMS_HOSTDEV void sentinel_to_indices_mult_imp(size_t ordinal, size_t index, size_t const *strides, size_t *indices, Rest... rest) {
    indices[index] = strides[index] * ordinal;

    sentinel_to_indices_mult_imp(ordinal, rest...);
}

template <typename Stride, typename Indices, typename... Rest>
void sentinel_to_indices_mult_imp(size_t ordinal, size_t index, Stride const &strides, Indices &indices, Rest... rest) {
    indices[index] = strides[index] * ordinal;

    sentinel_to_indices_mult_imp(ordinal, rest...);
}

/**
 * @brief Converts a single sentinel value into lists of indices.
 *
 * @param sentinel The sentinel to convert.
 * @param index_strides The strides of the unique indices that the sentinel is iterating over.
 * @param strides_inds The converted indices.
 */
template <size_t num_indices, typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
EINSUMS_HOSTDEV inline void sentinel_to_indices(size_t sentinel, size_t const *index_strides, StridesInds... strides_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ptrdiff_t i = 0; i < num_indices; i++) {
        size_t ordinal;
        if (index_strides[i] != 0) {
            ordinal = hold / index_strides[i];
            hold %= index_strides[i];
        } else {
            [[unlikely]] ordinal = 0;
        }

        sentinel_to_indices_mult_imp(ordinal, strides_inds...);
    }
}

template <size_t num_indices, typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
void sentinel_to_indices(size_t sentinel, std::array<size_t, num_indices> const &index_strides, StridesInds... strides_inds) {
    size_t hold = sentinel;

    auto args = std::forward_as_tuple(strides_inds...);

#pragma unroll
    for (ptrdiff_t i = 0; i < num_indices; i++) {
        size_t ordinal;
        if (index_strides[i] != 0) {
            ordinal = hold / index_strides[i];
            hold %= index_strides[i];
        } else {
            [[unlikely]] ordinal = 0;
        }

        sentinel_to_indices_mult_imp(ordinal, strides_inds...);
    }
}

template <typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
void sentinel_to_indices(size_t sentinel, StorageType const &index_strides, StridesInds... strides_inds) {
    size_t hold = sentinel;

    auto args = std::forward_as_tuple(strides_inds...);

    for (ptrdiff_t i = 0; i < index_strides.size(); i++) {
        size_t ordinal;
        if (index_strides[i] != 0) {
            ordinal = hold / index_strides[i];
            hold %= index_strides[i];
        } else {
            [[unlikely]] ordinal = 0;
        }

        sentinel_to_indices_mult_imp(ordinal, strides_inds...);
    }
}

/**
 * @brief The opposite of sentinel_to_indices. Calculates a sentinel given indices and strides.
 */
template <size_t num_unique_inds>
EINSUMS_HOSTDEV inline size_t indices_to_sentinel(size_t const *unique_strides, size_t const *inds) {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

template <size_t num_unique_inds>
inline size_t indices_to_sentinel(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                  std::array<size_t, num_unique_inds> const       &inds) {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

template <size_t num_unique_inds>
inline size_t indices_to_sentinel(std::array<size_t, num_unique_inds> const &unique_strides,
                                  std::array<size_t, num_unique_inds> const &inds) {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

namespace detail {

template <size_t index, size_t num_unique_inds, typename FirstIndex, typename... MultiIndex>
    requires(std::is_integral_v<std::decay_t<MultiIndex>> && ... && std::is_integral_v<std::decay_t<FirstIndex>>)
constexpr inline size_t indices_to_sentinel(std::array<std::int64_t, num_unique_inds> const &unique_strides, FirstIndex &&first_index,
                                            MultiIndex &&...indices) noexcept {
    if constexpr (sizeof...(MultiIndex) != 0) {
        return std::get<index>(unique_strides) * first_index +
               indices_to_sentinel<index + 1>(unique_strides, std::forward<MultiIndex>(indices)...);
    } else {
        return std::get<index>(unique_strides) * first_index;
    }
}
} // namespace detail

template <size_t num_unique_inds, typename... MultiIndex>
    requires(std::is_integral_v<std::decay_t<MultiIndex>> && ...)
constexpr inline size_t indices_to_sentinel(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                            MultiIndex &&...indices) noexcept {
    static_assert(sizeof...(MultiIndex) == num_unique_inds);
    return detail::indices_to_sentinel<0>(unique_strides, std::forward<MultiIndex>(indices)...);
}

template <typename StorageType1, typename StorageType2>
    requires(!std::is_integral_v<StorageType1> && !std::is_integral_v<StorageType2>)
inline size_t indices_to_sentinel(StorageType1 const &unique_strides, StorageType2 const &inds) {
    size_t out = 0;

    for (size_t i = 0; i < unique_strides.size(); i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

namespace detail {

template <size_t index, size_t num_unique_inds, typename FirstIndex, typename... MultiIndex>
    requires(std::is_integral_v<std::decay_t<MultiIndex>> && ... && std::is_integral_v<std::decay_t<FirstIndex>>)
inline size_t indices_to_sentinel_negative_check(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                                 std::array<int64_t, num_unique_inds> const &dims, FirstIndex &&first_index,
                                                 MultiIndex &&...indices) {
    auto const dim        = std::get<index>(dims);
    auto       temp_index = first_index;

    if constexpr (std::is_signed_v<FirstIndex>) {
        if (temp_index < 0) {
            temp_index += dim;
        }
    }

    if (temp_index < 0 || temp_index >= dim) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                "Index out of range! Index {} in rank {} was either greater than or equal to {} or less than {}",
                                first_index, index, dim, -(ptrdiff_t)dim);
    }
    if constexpr (sizeof...(MultiIndex) != 0) {
        return std::get<index>(unique_strides) * temp_index +
               indices_to_sentinel_negative_check<index + 1>(unique_strides, dims, std::forward<MultiIndex>(indices)...);
    } else {
        return std::get<index>(unique_strides) * temp_index;
    }
}
} // namespace detail

template <size_t num_unique_inds, typename... MultiIndex>
    requires(std::is_integral_v<std::decay_t<MultiIndex>> && ...)
inline size_t indices_to_sentinel_negative_check(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                                 std::array<int64_t, num_unique_inds> const      &dims, MultiIndex &&...indices) {
    static_assert(sizeof...(MultiIndex) == num_unique_inds);
    return detail::indices_to_sentinel_negative_check<0>(unique_strides, dims, std::forward<MultiIndex>(indices)...);
}

/**
 * @brief Converts indices to a sentinel. Checks for negative numbers and converts them.
 *
 * When checking for negative numbers, it will add the appropriate dimension to bring the index into
 * the range for that dimension. Can not be used on GPU since it throws errors. Also, running all those if-statements
 * would be very slow.
 */
template <typename StorageType1, typename StorageType2, typename StorageType3>
    requires requires {
        requires !std::is_arithmetic_v<StorageType1>;
        requires !std::is_arithmetic_v<StorageType2>;
        requires !std::is_arithmetic_v<StorageType3>;
    }
inline size_t indices_to_sentinel_negative_check(StorageType1 const &unique_strides, StorageType2 const &unique_dims,
                                                 StorageType3 const &inds) {
    size_t out = 0;

    if (inds.size() > unique_strides.size() || inds.size() > unique_dims.size()) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices supplied!");
    }

    for (size_t i = 0; i < inds.size(); i++) {
        ptrdiff_t ind = inds[i];

        if (ind < 0) [[unlikely]] {
            ind += unique_dims[i];
        }

        if (ind < 0 || ind >= unique_dims[i]) [[unlikely]] {
            EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                    "Index out of range! Index {} in rank {} was either greater than or equal to {} or less than {}",
                                    inds[i], i, unique_dims[i], -(ptrdiff_t)unique_dims[i]);
        }

        out += ind * unique_strides[i];
    }

    return out;
}

EINSUMS_EXPORT size_t dims_to_strides(std::vector<size_t> const &dims, std::vector<size_t> &out);

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 * @return The size calculated from the dimensions. Can be safely ignored.
 */
template <typename arr_type1, typename arr_type2, size_t Dims>
    requires(std::is_integral_v<arr_type1> && std::is_integral_v<arr_type2>)
constexpr size_t dims_to_strides(std::array<arr_type1, Dims> const &dims, std::array<arr_type2, Dims> &out) {
    size_t stride = 1;

    for (int i = Dims - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= dims[i];
    }

    return stride;
}

#ifndef DOXYGEN
template <int I, typename Head, typename Index>
int compile_index_table(std::tuple<Head> const &, Index const &, int &out) {
    if constexpr (std::is_same_v<Head, Index>) {
        out = I;
    } else {
        out = -1;
    }
    return 0;
}

template <int I, typename Head, typename... UniqueIndices, typename Index>
auto compile_index_table(std::tuple<Head, UniqueIndices...> const &, Index const &index, int &out)
    -> std::enable_if_t<sizeof...(UniqueIndices) != 0, int> {
    if constexpr (std::is_same_v<Head, Index>) {
        out = I;
    } else {
        compile_index_table<I + 1>(std::tuple<UniqueIndices...>(), index, out);
    }
    return 0;
}

template <typename... UniqueIndices, typename... Indices, size_t... I>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds, int *out,
                         std::index_sequence<I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<I>(to_inds), out[I])...};
}
#endif

/**
 * @brief Turn a list of indices into a link table.
 *
 * Takes a list of indices and creates a mapping so that an index list for a tensor can reference the unique index list.
 */
template <typename... UniqueIndices, typename... Indices>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds, int *out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}

#ifndef DOXYGEN
template <typename... UniqueIndices, typename... Indices, size_t... I>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds,
                         std::array<int, sizeof...(Indices)> &out, std::index_sequence<I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<I>(to_inds), out[I])...};
}

template <typename... UniqueIndices, typename... Indices>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds,
                         std::array<int, sizeof...(Indices)> &out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}
#endif
} // namespace einsums