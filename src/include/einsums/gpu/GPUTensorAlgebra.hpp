#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/GPUTensorAlgebra.hpp"

#include <bits/utility.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

namespace einsums {
namespace tensor_algebra {

namespace detail {

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
    return ::std::tuple{get_dim_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

__device__ inline bool is_zero(double value) {
    return value == 0.0;
}

__device__ inline bool is_zero(float value) {
    return value == 0.0f;
}

__device__ inline bool is_zero(hipComplex value) {
    return value.x == 0.0f && value.y == 0.0f;
}

__device__ inline bool is_zero(hipDoubleComplex value) {
    return value.x == 0.0 && value.y == 0.0;
}

__device__ inline void make_zero(double &value) {
    value = 0.0;
}

__device__ inline void make_zero(float &value) {
    value = 0.0f;
}

__device__ inline void make_zero(hipComplex &value) {
    value.x = 0.0f;
    value.y = 0.0f;
}

__device__ inline void make_zero(hipDoubleComplex &value) {
    value.x = 0.0;
    value.y = 0.0;
}

/**
 * @brief Converts a single sentinel value into a list of indices.
 */
template <size_t num_unique_inds>
__host__ __device__ inline void sentinel_to_indices(size_t sentinel, const size_t *unique_strides, size_t *out_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ssize_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            out_inds[i] = 0;
        }
    }
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(float *address, float value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(double *address, double value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(hipComplex *address, hipComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(hipDoubleComplex *address, hipDoubleComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

template <typename CDataType, typename ADataType, typename BDataType, size_t UniqueRank, size_t CRank, size_t ARank, size_t BRank>
__global__ void
einsum_generic_algorithm_gpu(const size_t *unique_strides, const int *C_index_table, const int *A_index_table, const int *B_index_table,
                             const CDataType C_prefactor, CDataType *C, const size_t *C_dims, const size_t *C_stride,
                             const ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const ADataType *A, const size_t *A_dims, const size_t *A_stride, const BDataType *B, const size_t *B_dims,
                             const size_t *B_stride, size_t max_index) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    ssize_t curr_index;

    size_t A_index[ARank], B_index[BRank], C_index[CRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel, C_sentinel;

    curr_index = thread_id;

    // First, set C.
    if (is_zero(C_prefactor)) {
        while (curr_index < C_dims[0] * C_stride[0]) {
            make_zero(C[curr_index]);
            curr_index += kernel_size;
        }
    } else {
        while (curr_index < C_dims[0] * C_stride[0]) {
            C[curr_index] *= C_prefactor;
            curr_index += kernel_size;
        }
    }

    __syncthreads();

    curr_index = thread_id;

    // Now, contract.
    while (curr_index < max_index) {
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

        atomicAdd_wrap(C + C_sentinel, (CDataType)(AB_prefactor * A[A_sentinel] * B[B_sentinel]));

        curr_index += kernel_size;
    }
}

/**
 * Compute kernel that runs when C has a rank of zero. There are some optimizations that can be made in this case.
 */
template <typename CDataType, typename ADataType, typename BDataType, size_t UniqueRank, size_t ARank, size_t BRank>
__global__ void
einsum_generic_zero_rank_gpu(const size_t *unique_strides, const int *A_index_table, const int *B_index_table, CDataType *C,
                             const ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const ADataType *A, const size_t *A_dims, const size_t *A_stride, const BDataType *B, const size_t *B_dims,
                             const size_t *B_stride, size_t max_index) {

    // Allocated by caller.
    extern __shared__ CDataType work[];

    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    // Clear the work array.
    make_zero(work[thread_id]);

    ssize_t curr_index;

    size_t A_index[ARank], B_index[BRank], Unique_index[UniqueRank];
    size_t A_sentinel, B_sentinel;

    curr_index = thread_id;

    while (curr_index < max_index) {
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

        work[thread_id] += A[A_sentinel] * B[B_sentinel];
    }

    atomicAdd_wrap(C, AB_prefactor * work[thread_id]);
}

template <typename... UniqueDims, size_t... I>
void dims_to_strides(const ::std::tuple<UniqueDims...> &dims, size_t *out, ::std::index_sequence<I...>) {
    ::std::array<size_t, sizeof...(UniqueDims)> arr{::std::get<I>(dims)...};

    size_t stride = 1;

    for (int i = sizeof...(UniqueDims) - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= arr[i];
    }
}

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 */
template <typename... UniqueDims>
void dims_to_strides(const ::std::tuple<UniqueDims...> &dims, size_t *out) {
    dims_to_strides(dims, out, ::std::make_index_sequence<sizeof...(UniqueDims)>());
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

template <typename... UniqueIndices, typename... CIndices, typename... AIndices, typename... BIndices, typename... UniqueDims,
          template <typename, size_t> typename CType, typename CDataType, size_t CRank, template <typename, size_t> typename AType,
          typename ADataType, size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
    requires requires {
        requires DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
    }
void einsum_generic_algorithm(const ::std::tuple<UniqueIndices...> &unique_indices, const ::std::tuple<CIndices...> &C_indices,
                              const ::std::tuple<AIndices...> &A_indices, const ::std::tuple<BIndices...> &B_indices,
                              const ::std::tuple<UniqueDims...> &unique_dims, const CDataType C_prefactor, CType<CDataType, CRank> *C,
                              const ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                              const AType<ADataType, ARank> &A, const BType<BDataType, BRank> &B) {
    using namespace einsums::gpu;

    size_t unique_strides[sizeof...(UniqueIndices)];

    dims_to_strides(unique_dims, unique_strides);

    int A_index_table[sizeof...(AIndices)], B_index_table[sizeof...(BIndices)], C_index_table[sizeof...(CIndices)];

    __device_ptr__ int    *A_index_table_gpu, *B_index_table_gpu, *C_index_table_gpu;
    __device_ptr__ size_t *unique_strides_gpu;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    hip_catch(hipMallocAsync((void **)&A_index_table_gpu, sizeof...(AIndices) * sizeof(int), get_stream()));
    hip_catch(hipMallocAsync((void **)&B_index_table_gpu, sizeof...(BIndices) * sizeof(int), get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipMallocAsync((void **)&C_index_table_gpu, sizeof...(CIndices) * sizeof(int), get_stream()));
    }
    hip_catch(hipMallocAsync((void **)&unique_strides_gpu, sizeof...(UniqueIndices) * sizeof(size_t), get_stream()));

    hip_catch(hipMemcpyAsync((void *)A_index_table_gpu, (const void *)A_index_table, sizeof...(AIndices) * sizeof(int),
                             hipMemcpyHostToDevice, get_stream()));
    hip_catch(hipMemcpyAsync((void *)B_index_table_gpu, (const void *)B_index_table, sizeof...(BIndices) * sizeof(int),
                             hipMemcpyHostToDevice, get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipMemcpyAsync((void *)C_index_table_gpu, (const void *)C_index_table, sizeof...(CIndices) * sizeof(int),
                                 hipMemcpyHostToDevice, get_stream()));
    }
    hip_catch(hipMemcpyAsync((void *)unique_strides_gpu, (const void *)unique_strides, sizeof...(UniqueIndices) * sizeof(size_t),
                             hipMemcpyHostToDevice, get_stream()));

    // Calculate the optimal launch bounds.
    dim3 threads = block_size(::std::get<0>(unique_dims) * unique_strides[0]),
         grid    = blocks(::std::get<0>(unique_dims) * unique_strides[0]);

    if constexpr (sizeof...(CIndices) != 0) {
        einsum_generic_algorithm_gpu<CDataType, ADataType, BDataType, sizeof...(UniqueIndices), CRank, ARank, BRank>
            <<<threads, grid, 0, get_stream()>>>(unique_strides_gpu, C_index_table_gpu, A_index_table_gpu, B_index_table_gpu, C_prefactor,
                                                 C->data(), C->gpu_dims(), C->gpu_strides(), AB_prefactor, A.data(), A.gpu_dims(),
                                                 A.gpu_strides(), B.data(), B.gpu_dims(), B.gpu_strides(),
                                                 ::std::get<0>(unique_dims) * unique_strides[0]);
    } else {
        // CDataType *work;
        // hip_catch(hipMalloc((void **)&work, threads.x * threads.y * threads.z * blocks.x * blocks.y * blocks.z * sizeof(CDataType)));
        if (C_prefactor == CDataType{0}) {
            *C = CDataType{0};
        } else {
            *C *= C_prefactor;
        }
        einsum_generic_zero_rank_gpu<CDataType, ADataType, BDataType, sizeof...(UniqueIndices), ARank, BRank>
            <<<threads, grid, threads.x * threads.y * threads.z * grid.x * grid.y * grid.z * sizeof(CDataType), get_stream()>>>(
                unique_strides_gpu, A_index_table_gpu, B_index_table_gpu, C->data(), AB_prefactor, A.data(), A.gpu_dims(), A.gpu_strides(),
                B.data(), B.gpu_dims(), B.gpu_strides(), ::std::get<0>(unique_dims) * unique_strides[0]);
    }

    hip_catch(hipFreeAsync(A_index_table_gpu, get_stream()));
    hip_catch(hipFreeAsync(B_index_table_gpu, get_stream()));
    if constexpr (sizeof...(CIndices) != 0) {
        hip_catch(hipFreeAsync(C_index_table_gpu, get_stream()));
    }
    hip_catch(hipFreeAsync(unique_strides_gpu, get_stream()));
}

template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
    }
auto einsum(const CDataType C_prefactor, const ::std::tuple<CIndices...> & /*Cs*/, CType<CDataType, CRank> *C,
            const ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
            const ::std::tuple<AIndices...> & /*As*/, const AType<ADataType, ARank> &A, const ::std::tuple<BIndices...> & /*Bs*/,
            const BType<BDataType, BRank> &B)
    -> ::std::enable_if_t<::std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>>
                              && ::std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>>
                                  && ::std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>>> {
    using namespace einsums::gpu;
    print::Indent const _indent;

    constexpr auto A_indices = ::std::tuple<AIndices...>();
    constexpr auto B_indices = ::std::tuple<BIndices...>();
    constexpr auto C_indices = ::std::tuple<CIndices...>();
    using ABDataType         = ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType>;

    // 1. Ensure the ranks are correct. (Compile-time check.)
    static_assert(sizeof...(CIndices) == CRank, "Rank of C does not match Indices given for C.");
    static_assert(sizeof...(AIndices) == ARank, "Rank of A does not match Indices given for A.");
    static_assert(sizeof...(BIndices) == BRank, "Rank of B does not match Indices given for B.");

    // 2. Determine the links from AIndices and BIndices
    constexpr auto linksAB = intersect_t<::std::tuple<AIndices...>, ::std::tuple<BIndices...>>();
    // 2a. Remove any links that appear in the target
    constexpr auto links = difference_t<decltype(linksAB), ::std::tuple<CIndices...>>();

    // 3. Determine the links between CIndices and AIndices
    constexpr auto CAlinks = intersect_t<::std::tuple<CIndices...>, ::std::tuple<AIndices...>>();

    // 4. Determine the links between CIndices and BIndices
    constexpr auto CBlinks = intersect_t<::std::tuple<CIndices...>, ::std::tuple<BIndices...>>();

    // Remove anything from A that exists in C
    constexpr auto CminusA = difference_t<::std::tuple<CIndices...>, ::std::tuple<AIndices...>>();
    constexpr auto CminusB = difference_t<::std::tuple<CIndices...>, ::std::tuple<BIndices...>>();

    constexpr bool have_remaining_indices_in_CminusA = ::std::tuple_size_v<decltype(CminusA)>;
    constexpr bool have_remaining_indices_in_CminusB = ::std::tuple_size_v<decltype(CminusB)>;

    // Determine unique indices in A
    constexpr auto A_only = difference_t<::std::tuple<AIndices...>, decltype(links)>();
    constexpr auto B_only = difference_t<::std::tuple<BIndices...>, decltype(links)>();

    constexpr auto A_unique    = unique_t<::std::tuple<AIndices...>>();
    constexpr auto B_unique    = unique_t<::std::tuple<BIndices...>>();
    constexpr auto C_unique    = unique_t<::std::tuple<CIndices...>>();
    constexpr auto All_unique  = unique_t<::std::tuple<CIndices..., AIndices..., BIndices...>>();
    constexpr auto link_unique = c_unique_t<decltype(links)>();

    constexpr bool A_hadamard_found = ::std::tuple_size_v<::std::tuple<AIndices...>> != ::std::tuple_size_v<decltype(A_unique)>;
    constexpr bool B_hadamard_found = ::std::tuple_size_v<::std::tuple<BIndices...>> != ::std::tuple_size_v<decltype(B_unique)>;
    constexpr bool C_hadamard_found = ::std::tuple_size_v<::std::tuple<CIndices...>> != ::std::tuple_size_v<decltype(C_unique)>;

    constexpr auto link_position_in_A    = ::einsums::tensor_algebra::detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B    = ::einsums::tensor_algebra::detail::find_type_with_position(link_unique, B_indices);
    constexpr auto link_position_in_link = ::einsums::tensor_algebra::detail::find_type_with_position(link_unique, links);

    constexpr auto target_position_in_A = ::einsums::tensor_algebra::detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B = ::einsums::tensor_algebra::detail::find_type_with_position(C_unique, B_indices);
    constexpr auto target_position_in_C = ::einsums::tensor_algebra::detail::find_type_with_position(C_unique, C_indices);

    constexpr auto A_target_position_in_C = ::einsums::tensor_algebra::detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C = ::einsums::tensor_algebra::detail::find_type_with_position(B_indices, C_indices);

    auto unique_target_dims = ::einsums::tensor_algebra::detail::get_dim_ranges_for(
        *C, ::einsums::tensor_algebra::detail::unique_find_type_with_position(C_unique, C_indices));
    auto unique_link_dims = ::einsums::tensor_algebra::detail::get_dim_ranges_for(A, link_position_in_A);
    auto unique_all_dims  = get_dim_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, All_unique);

    constexpr auto contiguous_link_position_in_A = ::einsums::tensor_algebra::detail::contiguous_positions(link_position_in_A);
    constexpr auto contiguous_link_position_in_B = ::einsums::tensor_algebra::detail::contiguous_positions(link_position_in_B);

    constexpr auto contiguous_target_position_in_A = ::einsums::tensor_algebra::detail::contiguous_positions(target_position_in_A);
    constexpr auto contiguous_target_position_in_B = ::einsums::tensor_algebra::detail::contiguous_positions(target_position_in_B);

    constexpr auto contiguous_A_targets_in_C = ::einsums::tensor_algebra::detail::contiguous_positions(A_target_position_in_C);
    constexpr auto contiguous_B_targets_in_C = ::einsums::tensor_algebra::detail::contiguous_positions(B_target_position_in_C);

    constexpr auto same_ordering_link_position_in_AB =
        ::einsums::tensor_algebra::detail::is_same_ordering(link_position_in_A, link_position_in_B);
    constexpr auto same_ordering_target_position_in_CA =
        ::einsums::tensor_algebra::detail::is_same_ordering(target_position_in_A, A_target_position_in_C);
    constexpr auto same_ordering_target_position_in_CB =
        ::einsums::tensor_algebra::detail::is_same_ordering(target_position_in_B, B_target_position_in_C);

    constexpr auto C_exactly_matches_A =
        sizeof...(CIndices) == sizeof...(AIndices) &&
        ::einsums::tensor_algebra::detail::same_indices<::std::tuple<CIndices...>, ::std::tuple<AIndices...>>();
    constexpr auto C_exactly_matches_B =
        sizeof...(CIndices) == sizeof...(BIndices) &&
        ::einsums::tensor_algebra::detail::same_indices<::std::tuple<CIndices...>, ::std::tuple<BIndices...>>();
    constexpr auto A_exactly_matches_B =
        ::einsums::tensor_algebra::detail::same_indices<::std::tuple<AIndices...>, ::std::tuple<BIndices...>>();

    constexpr auto is_gemm_possible = have_remaining_indices_in_CminusA && have_remaining_indices_in_CminusB &&
                                      contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                                      contiguous_target_position_in_B && contiguous_A_targets_in_C && contiguous_B_targets_in_C &&
                                      same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                                      same_ordering_target_position_in_CB && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    constexpr auto is_gemv_possible = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                                      same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                                      !same_ordering_target_position_in_CB && ::std::tuple_size_v<decltype(B_target_position_in_C)> == 0 &&
                                      !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    constexpr auto element_wise_multiplication =
        C_exactly_matches_A && C_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    constexpr auto dot_product =
        sizeof...(CIndices) == 0 && A_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    constexpr auto outer_product = ::std::tuple_size_v<decltype(linksAB)> == 0 && contiguous_target_position_in_A &&
                                   contiguous_target_position_in_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    // println("A_indices {}", print_tuple_no_type(A_indices));
    // println("B_indices {}", print_tuple_no_type(B_indices));
    // println("C_indices {}", print_tuple_no_type(C_indices));
    // println("A_unique {}", print_tuple_no_type(A_unique));
    // println("B_unique {}", print_tuple_no_type(B_unique));
    // println("C_unique {}", print_tuple_no_type(C_unique));
    // println("target_position_in_A {}", print_tuple_no_type(target_position_in_A));
    // println("target_position_in_B {}", print_tuple_no_type(target_position_in_B));
    // println("target_position_in_C {}", print_tuple_no_type(target_position_in_C));
    // println("link_position_in_A {}", print_tuple_no_type(link_position_in_A));
    // println("link_position_in_B {}", print_tuple_no_type(link_position_in_B));
    // println("contiguous_link_position_in_A {}", contiguous_link_position_in_A);
    // println("contiguous_link_position_in_B {}", contiguous_link_position_in_B);
    // println("contiguous_target_position_in_A {}", contiguous_target_position_in_A);
    // println("same_ordering_link_position_in_AB {}", same_ordering_link_position_in_AB);
    // println("same_ordering_target_position_in_CA {}", same_ordering_target_position_in_CA);
    // println("same_ordering_target_position_in_CB {}", same_ordering_target_position_in_CB);
    // println("std::tuple_size_v<decltype(B_target_position_in_C)> == 0 {}", std::tuple_size_v<decltype(B_target_position_in_C)> == 0);
    // println("A_hadamard_found {}", A_hadamard_found);
    // println("B_hadamard_found {}", B_hadamard_found);
    // println("C_hadamard_found {}", C_hadamard_found);

    // println("is_gemv_possible {}", is_gemv_possible);
    // println("is_gemm_possible {}", is_gemm_possible);
    // println("dot_product {}", dot_product);

    // Runtime check of sizes
#if defined(EINSUMS_RUNTIME_INDICES_CHECK)
    bool runtime_indices_abort{false};
    for_sequence<ARank>([&](auto a) {
        size_t dimA = A.dim(a);
        for_sequence<BRank>([&](auto b) {
            size_t dimB = B.dim(b);
            if (::std::get<a>(A_indices).letter == ::std::get<b>(B_indices).letter) {
                if (dimA != dimB) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(),
                            print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));
                    runtime_indices_abort = true;
                }
            }
        });
        for_sequence<CRank>([&](auto c) {
            size_t dimC = C->dim(c);
            if (::std::get<a>(A_indices).letter == ::std::get<c>(C_indices).letter) {
                if (dimA != dimC) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(),
                            print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));
                    runtime_indices_abort = true;
                }
            }
        });
    });
    for_sequence<BRank>([&](auto b) {
        size_t dimB = B.dim(b);
        for_sequence<CRank>([&](auto c) {
            size_t dimC = C->dim(c);
            if (::std::get<b>(B_indices).letter == ::std::get<c>(C_indices).letter) {
                if (dimB != dimC) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(),
                            print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));
                    runtime_indices_abort = true;
                }
            }
        });
    });

    if (runtime_indices_abort) {
        throw ::std::runtime_error("einsum: Inconsistent dimensions found!");
    }
#endif
    if constexpr (dot_product) {
        linear_algebra::dot(C_prefactor, *C, AB_prefactor, A, B);

        return;
    } else if constexpr (!::std::is_same_v<CDataType, ADataType> || !::std::is_same_v<CDataType, BDataType>) {
        // Mixed datatypes go directly to the generic algorithm.
        if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                      einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType> &&
                      einsums::detail::IsDeviceRankBlockTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
            if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
                throw std::runtime_error("einsums: Tensors need to have the same number of blocks.");
            }

            for (int i = 0; i < A.num_blocks(); i++) {
                if (A.block_dim(i) != B.block_dim(i) || A.block_dim(i) != C->block_dim(i) || B.block_dim(i) != C->block_dim(i)) {
                    throw new std::runtime_error("einsums: All blocks in the tensors need to have the same dimensions.");
                }
            }

#pragma omp task depend(in : A, B) depend(out : *C)
            {
                EINSUMS_OMP_PARALLEL_FOR
                for (int i = 0; i < A.num_blocks(); i++) {
                    ::einsums::tensor_algebra::detail::einsum_generic_algorithm(All_unique, C_indices, A_indices, B_indices,
                                                                                unique_all_dims, C_prefactor, &(C->block(i)), AB_prefactor,
                                                                                A[i], B[i]);
                    gpu::stream_wait();
                }
            }
        } else {
#pragma omp task depend(in : A, B) depend(out : *C)
            {
                ::einsums::tensor_algebra::detail::einsum_generic_algorithm(All_unique, C_indices, A_indices, B_indices, unique_all_dims,
                                                                            C_prefactor, C, AB_prefactor, A, B);
            }
        }
        return;
    } else if constexpr (element_wise_multiplication) {
        timer::GPUTimer const element_wise_multiplication{"element-wise multiplication"};

        auto target_dims = get_dim_ranges<CRank>(*C);
        auto view        = ::std::apply(ranges::views::cartesian_product, target_dims);

        // Ensure the various tensors passed in are the same dimensionality
        if (((C->dims() != A.dims()) || C->dims() != B.dims())) {
            println_abort("einsum: at least one tensor does not have same dimensionality as destination");
        }

        // The generic algorithm should work fine.
        if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                      einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType> &&
                      einsums::detail::IsDeviceRankBlockTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
            if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
                throw std::runtime_error("einsums: Tensors need to have the same number of blocks.");
            }

            for (int i = 0; i < A.num_blocks(); i++) {
                if (A.block_dim(i) != B.block_dim(i) || A.block_dim(i) != C->block_dim(i) || B.block_dim(i) != C->block_dim(i)) {
                    throw new std::runtime_error("einsums: All blocks in the tensors need to have the same dimensions.");
                }
            }

#pragma omp task depend(in : A, B) depend(out : *C)
            {
                EINSUMS_OMP_PARALLEL_FOR
                for (int i = 0; i < A.num_blocks(); i++) {
                    ::einsums::tensor_algebra::detail::einsum_generic_algorithm(All_unique, C_indices, A_indices, B_indices,
                                                                                unique_all_dims, C_prefactor, &(C->block(i)), AB_prefactor,
                                                                                A[i], B[i]);
                    gpu::stream_wait();
                }
            }
        } else {
#pragma omp task depend(in : A, B) depend(out : *C)
            {
                ::einsums::tensor_algebra::detail::einsum_generic_algorithm(All_unique, C_indices, A_indices, B_indices, unique_all_dims,
                                                                            C_prefactor, C, AB_prefactor, A, B);
            }
        }

        return;
        //     } else if constexpr (outer_product) {
        //         do { // do {} while (false) trick to allow us to use a break below to "break" out of the loop.
        //             constexpr bool swap_AB = ::std::get<1>(A_target_position_in_C) != 0;

        //             Dim<2> dC;
        //             dC[0] = product_dims(A_target_position_in_C, *C);
        //             dC[1] = product_dims(B_target_position_in_C, *C);
        //             if constexpr (swap_AB)
        //                 ::std::swap(dC[0], dC[1]);

        //             DeviceTensorView<CDataType, 2> tC{*C, dC};

        //             if (C_prefactor != CDataType{1.0})
        //                 linear_algebra::scale(C_prefactor, C);

        //             try {
        //                 if constexpr (swap_AB) {
        //                     linear_algebra::ger(AB_prefactor, B.to_rank_1_view(), A.to_rank_1_view(), &tC);
        //                 } else {
        //                     linear_algebra::ger(AB_prefactor, A.to_rank_1_view(), B.to_rank_1_view(), &tC);
        //                 }
        //             } catch (std::runtime_error &e) {
        // #if defined(EINSUMS_SHOW_WARNING)
        //                 println(
        //                     bg(fmt::color::yellow) | fg(fmt::color::black),
        //                     "Optimized outer product failed. Likely from a non-contiguous TensorView. Attempting to perform generic
        //                     algorithm.");
        // #endif
        //                 if (C_prefactor == CDataType{0.0}) {
        // #if defined(EINSUMS_SHOW_WARNING)
        //                     println(bg(fmt::color::red) | fg(fmt::color::white),
        //                             "WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor,
        //                             C->name());
        // #endif
        //                 } else {
        //                     linear_algebra::scale(1.0 / C_prefactor, C);
        //                 }
        //                 break; // out of the do {} while(false) loop.
        //             }
        //             // If we got to this position, assume we successfully called ger.
        //             return;
        //         } while (false);
    } else if constexpr (!OnlyUseGenericAlgorithm) {
        do { // do {} while (false) trick to allow us to use a break below to "break" out of the loop.
            if constexpr (is_gemv_possible) {
                if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
                    // Fall through to generic algorithm.
                    break;
                }

#pragma omp task depend(in : A, B) depend(out : *C)
                {

                    constexpr bool transpose_A = ::std::get<1>(link_position_in_A) == 0;

                    Dim<2>    dA;
                    Dim<1>    dB, dC;
                    Stride<2> sA;
                    Stride<1> sB, sC;

                    dA[0] = product_dims(A_target_position_in_C, *C);
                    dA[1] = product_dims(link_position_in_A, A);
                    sA[0] = last_stride(target_position_in_A, A);
                    sA[1] = last_stride(link_position_in_A, A);
                    if constexpr (transpose_A) {
                        ::std::swap(dA[0], dA[1]);
                        ::std::swap(sA[0], sA[1]);
                    }

                    dB[0] = product_dims(link_position_in_B, B);
                    sB[0] = last_stride(link_position_in_B, B);

                    dC[0] = product_dims(A_target_position_in_C, *C);
                    sC[0] = last_stride(A_target_position_in_C, *C);

                    const DeviceTensorView<ADataType, 2> tA{const_cast<AType<ADataType, ARank> &>(A), dA, sA};
                    const DeviceTensorView<BDataType, 1> tB{const_cast<BType<BDataType, BRank> &>(B), dB, sB};
                    DeviceTensorView<CDataType, 1>       tC{*C, dC, sC};

                    // println(*C);
                    // println(tC);
                    // println(A);
                    // println(tA);
                    // println(B);
                    // println(tB);

                    if constexpr (transpose_A) {
                        linear_algebra::gemv<true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                    } else {
                        linear_algebra::gemv<false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                    }
                }

                return;
            }
            // To use a gemm the input tensors need to be at least rank 2
            else if constexpr (CRank >= 2 && ARank >= 2 && BRank >= 2) {
                if constexpr (!A_hadamard_found && !B_hadamard_found && !C_hadamard_found) {
                    if constexpr (is_gemm_possible) {

                        if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
                            // Fall through to generic algorithm.
                            break;
                        }

#pragma omp task depend(in : A, B) depend(out : *C)
                        {

                            constexpr bool transpose_A = ::std::get<1>(link_position_in_A) == 0;
                            constexpr bool transpose_B = ::std::get<1>(link_position_in_B) != 0;
                            constexpr bool transpose_C = ::std::get<1>(A_target_position_in_C) != 0;

                            Dim<2>    dA, dB, dC;
                            Stride<2> sA, sB, sC;

                            dA[0] = ::einsums::tensor_algebra::detail::product_dims(A_target_position_in_C, *C);
                            dA[1] = ::einsums::tensor_algebra::detail::product_dims(link_position_in_A, A);
                            sA[0] = ::einsums::tensor_algebra::detail::last_stride(target_position_in_A, A);
                            sA[1] = ::einsums::tensor_algebra::detail::last_stride(link_position_in_A, A);
                            if constexpr (transpose_A) {
                                ::std::swap(dA[0], dA[1]);
                                ::std::swap(sA[0], sA[1]);
                            }

                            dB[0] = ::einsums::tensor_algebra::detail::product_dims(link_position_in_B, B);
                            dB[1] = ::einsums::tensor_algebra::detail::product_dims(B_target_position_in_C, *C);
                            sB[0] = ::einsums::tensor_algebra::detail::last_stride(link_position_in_B, B);
                            sB[1] = ::einsums::tensor_algebra::detail::last_stride(target_position_in_B, B);
                            if constexpr (transpose_B) {
                                ::std::swap(dB[0], dB[1]);
                                ::std::swap(sB[0], sB[1]);
                            }

                            dC[0] = ::einsums::tensor_algebra::detail::product_dims(A_target_position_in_C, *C);
                            dC[1] = ::einsums::tensor_algebra::detail::product_dims(B_target_position_in_C, *C);
                            sC[0] = ::einsums::tensor_algebra::detail::last_stride(A_target_position_in_C, *C);
                            sC[1] = ::einsums::tensor_algebra::detail::last_stride(B_target_position_in_C, *C);
                            if constexpr (transpose_C) {
                                ::std::swap(dC[0], dC[1]);
                                ::std::swap(sC[0], sC[1]);
                            }

                            DeviceTensorView<CDataType, 2>       tC{*C, dC, sC};
                            const DeviceTensorView<ADataType, 2> tA{const_cast<AType<ADataType, ARank> &>(A), dA, sA};
                            const DeviceTensorView<BDataType, 2> tB{const_cast<BType<BDataType, BRank> &>(B), dB, sB};

                            // println("--------------------");
                            // println(*C);
                            // println(tC);
                            // println("--------------------");
                            // println(A);
                            // println(tA);
                            // println("--------------------");
                            // println(B);
                            // println(tB);
                            // println("--------------------");

                            if constexpr (!transpose_C && !transpose_A && !transpose_B) {
                                linear_algebra::gemm<false, false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            } else if constexpr (!transpose_C && !transpose_A && transpose_B) {
                                linear_algebra::gemm<false, true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            } else if constexpr (!transpose_C && transpose_A && !transpose_B) {
                                linear_algebra::gemm<true, false>(AB_prefactor, tA, tB, C_prefactor, &tC);

                            } else if constexpr (!transpose_C && transpose_A && transpose_B) {
                                linear_algebra::gemm<true, true>(AB_prefactor, tA, tB, C_prefactor, &tC);

                            } else if constexpr (transpose_C && !transpose_A && !transpose_B) {
                                linear_algebra::gemm<true, true>(AB_prefactor, tB, tA, C_prefactor, &tC);

                            } else if constexpr (transpose_C && !transpose_A && transpose_B) {
                                linear_algebra::gemm<false, true>(AB_prefactor, tB, tA, C_prefactor, &tC);

                            } else if constexpr (transpose_C && transpose_A && !transpose_B) {
                                linear_algebra::gemm<true, false>(AB_prefactor, tB, tA, C_prefactor, &tC);

                            } else if constexpr (transpose_C && transpose_A && transpose_B) {
                                linear_algebra::gemm<false, false>(AB_prefactor, tB, tA, C_prefactor, &tC);

                            } else {
                                println("This GEMM case is not programmed: transpose_C {}, transpose_A {}, transpose_B {}", transpose_C,
                                        transpose_A, transpose_B);
                                ::std::abort();
                            }
                        }
                    }
                }
            }
            // If we make it here, then none of our algorithms for this last block could be used.
            // Fall through to the generic algorithm below.
        } while (false);
    }

    // If we somehow make it here, then none of our algorithms above could be used. Attempt to use
    // the generic algorithm instead.

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType> &&
                  einsums::detail::IsDeviceRankBlockTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
            throw std::runtime_error("einsums: Tensors need to have the same number of blocks.");
        }

        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) != B.block_dim(i) || A.block_dim(i) != C->block_dim(i) || B.block_dim(i) != C->block_dim(i)) {
                throw new std::runtime_error("einsums: All blocks in the tensors need to have the same dimensions.");
            }
        }

#pragma omp task depend(in : A, B) depend(out : *C)
        {
            EINSUMS_OMP_PARALLEL_FOR
            for (int i = 0; i < A.num_blocks(); i++) {
                ::einsums::tensor_algebra::detail::einsum_generic_algorithm(All_unique, C_indices, A_indices, B_indices, unique_all_dims,
                                                                            C_prefactor, &(C->block(i)), AB_prefactor, A[i], B[i]);
                gpu::stream_wait();
            }
        }
    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                         einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType> &&
                         !einsums::detail::IsDeviceRankBlockTensorV<CType<CDataType, CRank>, CRank, CDataType> && CRank == 0) {
#pragma omp task depend(in : A, B) depend(out : *C)
        {
            ::einsums::tensor_algebra::detail::einsum_generic_algorithm(All_unique, C_indices, A_indices, B_indices, unique_all_dims,
                                                                        C_prefactor, C, AB_prefactor, (DeviceTensor<ADataType, ARank>)A,
                                                                        (DeviceTensor<BDataType, BRank>)B);
        }
    } else {
#pragma omp task depend(in : A, B) depend(out : *C)
        {
            ::einsums::tensor_algebra::detail::einsum_generic_algorithm(All_unique, C_indices, A_indices, B_indices, unique_all_dims,
                                                                        C_prefactor, C, AB_prefactor, A, B);
        }
    }
}

} // namespace detail

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank, template <typename, size_t> typename BType,
          typename BDataType, size_t BRank, template <typename, size_t> typename CType, typename CDataType, size_t CRank,
          typename... CIndices, typename... AIndices, typename... BIndices, typename U>
    requires requires {
        requires DeviceRankTensor<CType<CDataType, CRank>, CRank, CDataType>;
        requires DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
    }
auto einsum(const U UC_prefactor, const ::std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C, const U UAB_prefactor,
            const ::std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A, const ::std::tuple<BIndices...> &B_indices,
            const BType<BDataType, BRank> &B)
    -> ::std::enable_if_t<::std::is_base_of_v<::einsums::detail::TensorBase<ADataType, ARank>, AType<ADataType, ARank>>
                              && ::std::is_base_of_v<::einsums::detail::TensorBase<BDataType, BRank>, BType<BDataType, BRank>>
                                  && ::std::is_base_of_v<::einsums::detail::TensorBase<CDataType, CRank>, CType<CDataType, CRank>>
                                      && ::std::is_arithmetic_v<U>> {
    using ABDataType = ::std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType>;

    LabeledSection1(FP_ZERO != ::std::fpclassify(UC_prefactor)
                        ? fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices),
                                      UAB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices),
                                      UC_prefactor, C->name(), print_tuple_no_type(C_indices))
                        : fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{})", C->name(), print_tuple_no_type(C_indices), UAB_prefactor,
                                      A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices)));

    const CDataType  C_prefactor  = UC_prefactor;
    const ABDataType AB_prefactor = UAB_prefactor;

#if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    // Clone C into a new tensor
    DeviceTensor<CDataType, CRank> testC{C->dims()};
    testC = (DeviceTensor<CDataType, CRank>)*C;

    // Perform the einsum using only the generic algorithm
    timer::GPUTimer *testing = new timer::GPUTimer("testing");
    ::einsums::tensor_algebra::detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, B);
    delete testing;
#endif

    // // If the tensors are all block tensors, handle them appropriately.
    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType> &&
                  einsums::detail::IsDeviceRankBlockTensorV<CType<CDataType, CRank>, CRank, CDataType>) {
        ::einsums::tensor_algebra::detail::einsum<false>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);

        //         if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
        //             throw std::runtime_error("All tensors passed to einsums need to have the same number of blocks.");
        //         }

        //         for (int i = 0; i < A.num_blocks(); i++) {
        //             if (A.block_dim(i) != B.block_dim(i) || A.block_dim(i) != C->block_dim(i) || B.block_dim(i) != C->block_dim(i)) {
        //                 throw std::runtime_error("Inconsistent block sizes in tensors.");
        //             }
        //         }

        //         // Perform einsum on each block separately.
        //         for (int i = 0; i < A.num_blocks(); i++) {
        //             if(A.block_dim(i) == 0) {
        //                 continue;
        //             }
        //             ::einsums::tensor_algebra::detail::einsum<false>(C_prefactor, C_indices, &(C->block(i)), AB_prefactor, A_indices,
        //             A[i], B_indices, B[i]);
        //         }
        //     } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
        //                          einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType> && CRank == 0) {
        //         if (A.num_blocks() != B.num_blocks()) {
        //             throw std::runtime_error("All tensors passed to einsums need to have the same number of blocks.");
        //         }

        //         for (int i = 0; i < A.num_blocks(); i++) {
        //             if (A.block_dim(i) != B.block_dim(i)) {
        //                 throw std::runtime_error("Inconsistent block sizes in tensors.");
        //             }
        //         }

        //         // Perform einsum on each block separately.
        //         if(C_prefactor == CDataType{0}) {
        //             *C = 0;
        //         } else {
        //             *C *= C_prefactor;
        //         }

        //         for (int i = 0; i < A.num_blocks(); i++) {
        //             if(A.block_dim(i) == 0) {
        //                 continue;
        //             }
        //             DeviceTensor<CDataType, 0> temp;

        //             temp = 0;

        //             ::einsums::tensor_algebra::detail::einsum<false>(C_prefactor, C_indices, &temp, AB_prefactor, A_indices, A[i],
        //             B_indices, B[i]); *C += temp;
        //         }
        //     } else
    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> ||
                         einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        // Use generic algorithm if mixing block and normal tensors.
        ::einsums::tensor_algebra::detail::einsum<true>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
    } else {
        // Default einsums.
        ::einsums::tensor_algebra::detail::einsum<false>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
    }

#if 0 // defined(EINSUMS_TEST_NANS)
    if constexpr (CRank != 0) {
        auto target_dims = get_dim_ranges<CRank>(*C);
        for (auto target_combination : ::std::apply(ranges::views::cartesian_product, target_dims)) {
            CDataType Cvalue{::std::apply(*C, target_combination)};
            if constexpr (!IsComplexV<CDataType>) {
                if (::std::isnan(Cvalue)) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "NaN DETECTED!");
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor,
                            C->name(), print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    throw ::std::runtime_error("NAN detected in resulting tensor.");
                }

                if (::std::isinf(Cvalue)) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "Infinity DETECTED!");
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor,
                            C->name(), print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    throw ::std::runtime_error("Infinity detected in resulting tensor.");
                }

                if (::std::abs(Cvalue) > 100000000) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "Large value DETECTED!");
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor,
                            C->name(), print_tuple_no_type(C_indices), AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(),
                            print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    throw ::std::runtime_error("Large value detected in resulting tensor.");
                }
            }
        }
    }
#endif

#if 0 // defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    if constexpr (CRank != 0) {
        // Need to walk through the entire C and testC comparing values and reporting differences.
        auto target_dims = get_dim_ranges<CRank>(*C);
        bool print_info_and_abort{false};

        for (auto target_combination : ::std::apply(ranges::views::cartesian_product, target_dims)) {
            CDataType Cvalue{::std::apply(*C, target_combination)};
            CDataType Ctest{::std::apply(testC, target_combination)};

            if constexpr (!IsComplexV<CDataType>) {
                if (::std::isnan(Cvalue) || ::std::isnan(Ctest)) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "NAN DETECTED!");
                    println("Source tensors");
                    println(A);
                    println(B);
                    if (::std::isnan(Cvalue)) {
                        println("NAN detected in C");
                        println("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(*C);
                    }
                    if (::std::isnan(Ctest)) {
                        println("NAN detected in reference Ctest");
                        println("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(testC);
                    }

                    print_info_and_abort = true;
                }
            }

#    if defined(EINSUMS_USE_CATCH2)
            if constexpr (!IsComplexV<CDataType>) {
                REQUIRE_THAT(Cvalue,
                             Catch::Matchers::WithinRel(Ctest, static_cast<CDataType>(0.001)) || Catch::Matchers::WithinAbs(0, 0.0001));
                CHECK(print_info_and_abort == false);
            }
#    endif

            if (::std::abs(Cvalue - Ctest) > 1.0E-6) {
                print_info_and_abort = true;
            }

            if (print_info_and_abort) {
                println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "    !!! EINSUM ERROR !!!");
                if constexpr (IsComplexV<CDataType>) {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f} + {:20.14f}i", Cvalue.real(),
                            Cvalue.imag());
                } else {
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                    println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);
                }
                println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ({:})", print_tuple_no_type(target_combination));
                ::std::string C_prefactor_string;
                if constexpr (IsComplexV<CDataType>) {
                    C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
                } else {
                    C_prefactor_string = fmt::format("{:f}", C_prefactor);
                }
                ::std::string AB_prefactor_string;
                if constexpr (IsComplexV<ABDataType>) {
                    AB_prefactor_string = fmt::format("({:f} + {:f}i)", AB_prefactor.real(), AB_prefactor.imag());
                } else {
                    AB_prefactor_string = fmt::format("{:f}", AB_prefactor);
                }
                println(bg(fmt::color::red) | fg(fmt::color::white), "    {} C({:}) += {:f} A({:}) * B({:})", C_prefactor_string,
                        print_tuple_no_type(C_indices), AB_prefactor_string, print_tuple_no_type(A_indices),
                        print_tuple_no_type(B_indices));

                println("Expected:");
                println(testC);
                println("Obtained");
                println(*C);
                println(A);
                println(B);
#    if defined(EINSUMS_TEST_EINSUM_ABORT)
                ::std::abort();
#    endif
            }
        }
    } else {
        const CDataType Cvalue = *C;
        const CDataType Ctest  = testC;

        // testC could be a Tensor<CDataType, 0> type. Cast it to the underlying data type.
        if (::std::abs(Cvalue - (CDataType)testC) > 1.0E-6) {
            println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "!!! EINSUM ERROR !!!");
            if constexpr (IsComplexV<CDataType>) {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f} + {:20.14f}i", Cvalue.real(), Cvalue.imag());
            } else {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);
            }

            println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ()");
            ::std::string C_prefactor_string;
            if constexpr (IsComplexV<CDataType>) {
                C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
            } else {
                C_prefactor_string = fmt::format("{:f}", C_prefactor);
            }
            ::std::string AB_prefactor_string;
            if constexpr (IsComplexV<ABDataType>) {
                AB_prefactor_string = fmt::format("({:f} + {:f}i)", AB_prefactor.real(), AB_prefactor.imag());
            } else {
                AB_prefactor_string = fmt::format("{:f}", AB_prefactor);
            }
            println(bg(fmt::color::red) | fg(fmt::color::white), "    {} C() += {} A({:}) * B({:})", C_prefactor_string,
                    AB_prefactor_string, print_tuple_no_type(A_indices), print_tuple_no_type(B_indices));

            println("Expected:");
            println(testC);
            println("Obtained");
            println(*C);
            println(A);
            println(B);

#    if defined(EINSUMS_TEST_EINSUM_ABORT)
            ::std::abort();
#    endif
        }
    }
#endif
}

/** Computes the Khatri-Rao product of tensors A and B.
 *
 * Example:
 *    Tensor<2> result = khatri_rao(Indices{I, r}, A, Indices{J, r}, B);
 *
 * Result is described as {(I,J), r}. If multiple common indices are provided they will be collapsed into a single index in the result.
 */
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank,
          typename... AIndices, typename... BIndices, typename T = double>
auto khatri_rao(const ::std::tuple<AIndices...> &, const AType<T, ARank> &A, const ::std::tuple<BIndices...> &, const BType<T, BRank> &B)
    -> ::std::enable_if_t<::std::is_base_of_v<::einsums::detail::TensorBase<T, ARank>, AType<T, ARank>>
                              && ::std::is_base_of_v<::einsums::detail::TensorBase<T, BRank>, BType<T, BRank>>,
                          DeviceTensor<T, 2>> {
    LabeledSection0();

    constexpr auto A_indices = ::std::tuple<AIndices...>();
    constexpr auto B_indices = ::std::tuple<BIndices...>();

    // Determine the common indices between A and B
    constexpr auto common = intersect_t<::std::tuple<AIndices...>, ::std::tuple<BIndices...>>();
    // Determine unique indices in A
    constexpr auto A_only = difference_t<::std::tuple<AIndices...>, decltype(common)>();
    // Determine unique indices in B
    constexpr auto B_only = difference_t<::std::tuple<BIndices...>, decltype(common)>();

    // Record the positions of each types.
    constexpr auto A_common_position = ::einsums::tensor_algebra::detail::find_type_with_position(common, A_indices);
    constexpr auto B_common_position = ::einsums::tensor_algebra::detail::find_type_with_position(common, B_indices);
    constexpr auto A_only_position   = ::einsums::tensor_algebra::detail::find_type_with_position(A_only, A_indices);
    constexpr auto B_only_position   = ::einsums::tensor_algebra::detail::find_type_with_position(B_only, B_indices);

    // Obtain dimensions of the indices discovered above
    auto A_common_dims = ::einsums::tensor_algebra::detail::get_dim_for(A, A_common_position);
    auto B_common_dims = ::einsums::tensor_algebra::detail::get_dim_for(B, B_common_position);
    auto A_only_dims   = ::einsums::tensor_algebra::detail::get_dim_for(A, A_only_position);
    auto B_only_dims   = ::einsums::tensor_algebra::detail::get_dim_for(B, B_only_position);

    // Sanity check - ensure the common dims between A and B are the same size.
    for_sequence<::std::tuple_size_v<decltype(common)>>([&](auto i) {
        if (::std::get<i>(A_common_dims) != ::std::get<i>(B_common_dims)) {
            throw ::std::runtime_error(fmt::format("Common dimensions for index {} of A and B do not match.", ::std::get<i>(common)));
        }
    });

    auto result_dims = ::std::tuple_cat(std::make_tuple("KR product"), A_only_dims, B_only_dims, A_common_dims);

    // Construct resulting tensor
    auto result = ::std::make_from_tuple<DeviceTensor<T, ::std::tuple_size_v<decltype(result_dims)> - 1>>(result_dims);

    // Perform the actual Khatri-Rao product using our einsum routine.
    ::einsums::tensor_algebra::einsum(::std::tuple_cat(A_only, B_only, common), &result, ::std::tuple_cat(A_only, common), A,
                                      ::std::tuple_cat(B_only, common), B);

    // Return a reconstruction of the result tensor ... this can be considered as a simple reshape of the tensor.
    return DeviceTensor<T, 2>{::std::move(result), "KR product", -1, ::einsums::tensor_algebra::detail::product_dims(A_common_position, A)};
}

} // namespace tensor_algebra
} // namespace einsums