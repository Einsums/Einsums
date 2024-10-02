#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

namespace einsums::tensor_algebra::detail {

#if defined(__HIP__) && !defined(HOSTDEV)
#    define HOSTDEV __host__ __device__
#else
#    define HOSTDEV
#endif

template <typename UniqueIndex, int BDim, typename BType>
inline size_t get_dim_ranges_for_many_b(const BType &B, const ::std::tuple<> &B_indices) {
    return 1;
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead>
inline auto get_dim_ranges_for_many_b(const BType &B, const ::std::tuple<BHead> &B_indices)
    -> ::std::enable_if<::std::is_same_v<BHead, UniqueIndex>, size_t> {
    return B.dim(BDim);
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_b(const BType &B, const ::std::tuple<BHead, BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<BHead, UniqueIndex>) {
        return B.dim(BDim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, BDim + 1>(B, ::std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(const AType &A, const ::std::tuple<> &A_indices, const BType &B,
                                        const ::std::tuple<BIndices...> &B_indices) {
    return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(const AType &A, const ::std::tuple<AHead> &A_indices, const BType &B,
                                        const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_a(const AType &A, const ::std::tuple<AHead, AIndices...> &A_indices, const BType &B,
                                      const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(AIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, ADim + 1>(A, ::std::tuple<AIndices...>(), B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename... AIndices, typename... BIndices>
inline size_t get_dim_ranges_for_many_c(const CType &C, const ::std::tuple<> &C_indices, const AType &A,
                                        const ::std::tuple<AIndices...> &A_indices, const BType &B,
                                        const ::std::tuple<BIndices...> &B_indices) {
    return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... AIndices,
          typename... BIndices>
inline size_t get_dim_ranges_for_many_c(const CType &C, const ::std::tuple<CHead> &C_indices, const AType &A,
                                        const ::std::tuple<AIndices...> &A_indices, const BType &B,
                                        const ::std::tuple<BIndices...> &B_indices) {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        return C.dim(CDim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... CIndices,
          typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_c(const CType &C, const ::std::tuple<CHead, CIndices...> &C_indices, const AType &A,
                                      const ::std::tuple<AIndices...> &A_indices, const BType &B,
                                      const ::std::tuple<BIndices...> &B_indices) -> ::std::enable_if_t<sizeof...(CIndices) != 0, size_t> {
    if constexpr (::std::is_same_v<CHead, UniqueIndex>) {
        return C.dim(CDim);
    } else {
        return get_dim_ranges_for_many_c<UniqueIndex, CDim + 1>(C, ::std::tuple<CIndices...>(), A, A_indices, B, B_indices);
    }
}

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
inline auto get_dim_ranges_for_many(const CType &C, const ::std::tuple<CIndices...> &C_indices, const AType &A,
                                    const ::std::tuple<AIndices...> &A_indices, const BType &B, const ::std::tuple<BIndices...> &B_indices,
                                    const ::std::tuple<AllUniqueIndices...> &All_unique_indices) {
    return ::std::array{get_dim_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

/**
 * @brief Converts a single sentinel value into a list of indices.
 *
 * @param sentinel The sentinel to convert.
 * @param unique_strides The strides of the unique indices that the sentinel is iterating over.
 * @param out_inds The converted indices.
 */
template <size_t num_unique_inds>
HOSTDEV inline void sentinel_to_indices(size_t sentinel, const size_t *unique_strides, size_t *out_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ssize_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            [[unlikely]] out_inds[i] = 0;
        }
    }
}

template <size_t num_unique_inds>
inline void sentinel_to_indices(size_t sentinel, const std::array<size_t, num_unique_inds> &unique_strides,
                                std::array<size_t, num_unique_inds> &out_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ssize_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            [[unlikely]] out_inds[i] = 0;
        }
    }
}

template <typename StorageType1, typename StorageType2>
inline void sentinel_to_indices(size_t sentinel, const StorageType1 &unique_strides, StorageType2 &out_inds) {
    size_t hold = sentinel;

    if (out_inds.size() != unique_strides.size()) {
        out_inds.resize(unique_strides.size());
    }

    for (ssize_t i = 0; i < unique_strides.size(); i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            [[unlikely]] out_inds[i] = 0;
        }
    }
}

/**
 * @brief The opposite of sentinel_to_indices. Calculates a sentinel given indices and strides.
 */
template <size_t num_unique_inds>
HOSTDEV inline size_t indices_to_sentinel(const size_t *unique_strides, const size_t *inds) {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

template <size_t num_unique_inds>
inline size_t indices_to_sentinel(const std::array<size_t, num_unique_inds> &unique_strides,
                                  const std::array<size_t, num_unique_inds> &inds) {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

template <typename StorageType1, typename StorageType2>
inline size_t indices_to_sentinel(const StorageType1 &unique_strides, const StorageType2 &inds) {
    size_t out = 0;

    for (size_t i = 0; i < unique_strides.size(); i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

/**
 * @brief Converts indices to a sentinel. Checks for negative numbers and converts them.
 *
 * When checking for negative numbers, it will add the appropriate dimension to bring the index into
 * the range for that dimension. Can not be used on GPU since it throws errors. Also, running all those if-statements
 * would be very slow.
 */
template <size_t num_unique_inds>
inline size_t indices_to_sentinel_negative_check(const size_t *unique_strides, const size_t *unique_dims, const size_t *inds) {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        size_t ind = inds[i];

        if (ind < 0) {
            [[unlikely]] ind += unique_dims[i];
        }

        if (ind < 0 || ind >= unique_dims[i]) {
            [[unlikely]] throw EINSUMSEXCEPTION(
                fmt::format("Index out of range! Index {} in rank {} was either greater than or equal to {} or less than {}", inds[i], i,
                            unique_dims[i], -unique_dims[i]));
        }

        out += ind * unique_strides[i];
    }

    return out;
}

template <size_t num_unique_inds>
inline size_t indices_to_sentinel_negative_check(const std::array<size_t, num_unique_inds> &unique_strides,
                                                 const std::array<size_t, num_unique_inds> &unique_dims,
                                                 const std::array<size_t, num_unique_inds> &inds) {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        size_t ind = inds[i];

        if (ind < 0) {
            [[unlikely]] ind += unique_dims[i];
        }

        if (ind < 0 || ind >= unique_dims[i]) {
            [[unlikely]] throw EINSUMSEXCEPTION(
                fmt::format("Index out of range! Index {} in rank {} was either greater than or equal to {} or less than {}", inds[i], i,
                            unique_dims[i], -unique_dims[i]));
        }

        out += ind * unique_strides[i];
    }

    return out;
}

template <typename StorageType1, typename StorageType2, typename StorageType3>
    requires requires {
        requires !std::is_arithmetic_v<StorageType1>;
        requires !std::is_arithmetic_v<StorageType2>;
        requires !std::is_arithmetic_v<StorageType3>;
    }
inline size_t indices_to_sentinel_negative_check(const StorageType1 &unique_strides, const StorageType2 &unique_dims,
                                                 const StorageType3 &inds) {
    size_t out = 0;

    for (size_t i = 0; i < inds.size(); i++) {
        size_t ind = inds[i];

        if (ind < 0) {
            [[unlikely]] ind += unique_dims[i];
        }

        if (ind < 0 || ind >= unique_dims[i]) {
            [[unlikely]] throw EINSUMSEXCEPTION(
                fmt::format("Index out of range! Index {} in rank {} was either greater than or equal to {} or less than {}", inds[i], i,
                            unique_dims[i], -unique_dims[i]));
        }

        out += ind * unique_strides[i];
    }

    return out;
}

EINSUMS_EXPORT void dims_to_strides(const std::vector<size_t> &dims, std::vector<size_t> &out);

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 */
template <size_t Dims>
void dims_to_strides(const ::std::array<size_t, Dims> &dims, std::array<size_t, Dims> &out) {
    size_t stride = 1;

    for (int i = Dims - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= dims[i];
    }
}

template <size_t Dims>
void dims_to_strides(const Dim<Dims> &dims, std::array<size_t, Dims> &out) {
    size_t stride = 1;

    for (int i = Dims - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= dims[i];
    }
}

template <int I, typename Head, typename Index>
int compile_index_table(const ::std::tuple<Head> &, const Index &, int &out) {
    if constexpr (::std::is_same_v<Head, Index>) {
        out = I;
    } else {
        out = -1;
    }
    return 0;
}

template <int I, typename Head, typename... UniqueIndices, typename Index>
auto compile_index_table(const ::std::tuple<Head, UniqueIndices...> &, const Index &index,
                         int &out) -> ::std::enable_if_t<sizeof...(UniqueIndices) != 0, int> {
    if constexpr (::std::is_same_v<Head, Index>) {
        out = I;
    } else {
        compile_index_table<I + 1>(::std::tuple<UniqueIndices...>(), index, out);
    }
    return 0;
}

template <typename... UniqueIndices, typename... Indices, size_t... I>
void compile_index_table(const ::std::tuple<UniqueIndices...> &from_inds, const ::std::tuple<Indices...> &to_inds, int *out,
                         ::std::index_sequence<I...>) {
    ::std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, ::std::get<I>(to_inds), out[I])...};
}

/**
 * @brief Turn a list of indices into a link table.
 *
 * Takes a list of indices and creates a mapping so that an index list for a tensor can reference the unique index list.
 */
template <typename... UniqueIndices, typename... Indices>
void compile_index_table(const ::std::tuple<UniqueIndices...> &from_inds, const ::std::tuple<Indices...> &to_inds, int *out) {
    compile_index_table(from_inds, to_inds, out, ::std::make_index_sequence<sizeof...(Indices)>());
}

template <typename... UniqueIndices, typename... Indices, size_t... I>
void compile_index_table(const ::std::tuple<UniqueIndices...> &from_inds, const ::std::tuple<Indices...> &to_inds,
                         std::array<int, sizeof...(Indices)> &out, ::std::index_sequence<I...>) {
    ::std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, ::std::get<I>(to_inds), out[I])...};
}

template <typename... UniqueIndices, typename... Indices>
void compile_index_table(const ::std::tuple<UniqueIndices...> &from_inds, const ::std::tuple<Indices...> &to_inds,
                         std::array<int, sizeof...(Indices)> &out) {
    compile_index_table(from_inds, to_inds, out, ::std::make_index_sequence<sizeof...(Indices)>());
}
} // namespace einsums::tensor_algebra::detail