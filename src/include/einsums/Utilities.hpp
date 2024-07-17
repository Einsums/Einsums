//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#ifdef __HIP__
#    include "einsums/DeviceTensor.hpp"
#endif
#include "einsums/OpenMP.h"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/ComplexTraits.hpp"

#ifdef EINSUMS_USE_CATCH2
#    include <catch2/catch_all.hpp>
#endif

#include <numbers>

// Forward definitions for positive definite matrices.
namespace einsums::linear_algebra {
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T, typename U>
    requires requires {
        requires InSamePlace<AType<T, Rank>, BType<T, Rank>, 2, 2, T, T>;
        requires InSamePlace<AType<T, Rank>, CType<T, Rank>, 2, 2, T, T>;
        requires std::convertible_to<U, T>;
    }
void gemm(const U alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const U beta, CType<T, Rank> *C);

template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
    requires CoreRankTensor<TensorType<T, TensorRank>, 2, T>
auto getrf(TensorType<T, TensorRank> *A, std::vector<blas_int> *pivot) -> int;

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
auto qr(const AType<T, ARank> &_A) -> std::tuple<Tensor<T, 2>, Tensor<T, 1>>;

template <typename T>
auto q(const Tensor<T, 2> &qr, const Tensor<T, 1> &tau) -> Tensor<T, 2>;
} // namespace einsums::linear_algebra

namespace einsums {

/**
 * @brief Create a new tensor with \p name and \p index filled with incremental data.
 *
 * Just a simple factory function for creating new tensor with initial data. The first element of the tensor is 0 and then each subsequent
 * entry is +1.0 of the prior element. Defaults to using double for the underlying data and automatically determines the rank of the tensor
 * from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2> with data ranging from 0.0 to 8.0
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3> with dat ranging from 0.0f to 119.0f
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param name The name of the new tensor.
 * @param index The arguments needed to construct the tensor.
 * @return A new tensor filled with incremented data
 */
template <typename T = double, typename... MultiIndex>
auto create_incremented_tensor(const std::string &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    Tensor<T, sizeof...(MultiIndex)> A(name, std::forward<MultiIndex>(index)...);

    T    counter{0.0};
    auto target_dims = get_dim_ranges<sizeof...(MultiIndex)>(A);
    auto view        = std::apply(ranges::views::cartesian_product, target_dims);

    for (auto it = view.begin(); it != view.end(); it++) {
        std::apply(A, *it) = counter;
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            counter += T{1.0, 1.0};
        } else {
            counter += T{1.0};
        }
    }

    return A;
}

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam Normalize Should the resulting random data be normalized. Defaults to false.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param name The name of the new tensor.
 * @param index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 */
template <typename T = double, bool Normalize = false, typename... MultiIndex>
auto create_random_tensor(const std::string &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    Section const section{fmt::format("create_random_tensor {}", name)};

    Tensor<T, sizeof...(MultiIndex)> A(name, std::forward<MultiIndex>(index)...);

    double const lower_bound = -1.0;
    double const upper_bound = 1.0;

    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine             re;

    if constexpr (std::is_same_v<T, std::complex<float>>) {
#pragma omp parallel default(none) shared(A, re, unif)
        {
            auto tid       = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin     = A.vector_data().begin() + chunksize * tid;
            auto end       = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(),
                          [&]() { return T{static_cast<float>(unif(re)), static_cast<float>(unif(re))}; });
        }
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
#pragma omp parallel default(none) shared(A, re, unif)
        {
            auto tid       = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin     = A.vector_data().begin() + chunksize * tid;
            auto end       = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(),
                          [&]() { return T{static_cast<double>(unif(re)), static_cast<double>(unif(re))}; });
        }
    } else {
#pragma omp parallel default(none) shared(A, re, unif)
        {
            auto tid       = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin     = A.vector_data().begin() + chunksize * tid;
            auto end       = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() { return static_cast<T>(unif(re)); });
        }
    }

    if constexpr (Normalize && sizeof...(MultiIndex) == 2) {
        for (int col = 0; col < A.dim(-1); col++) {
            RemoveComplexT<T> scale{1};
            RemoveComplexT<T> sumsq{0};

            auto column = A(All, col);
            // auto collapsed = TensorView{A, Dim<2>{-1, A.dim(-1)}};
            // auto column = collapsed(All, col);
            sum_square(column, &scale, &sumsq);
            T value = scale * sqrt(sumsq);
            column /= value;
        }
    }

    return A;
}

#ifdef __HIP__
template <typename T = double, bool Normalize = false, typename... MultiIndex>
auto create_random_gpu_tensor(const std::string &name, MultiIndex... index) -> DeviceTensor<T, sizeof...(MultiIndex)> {
    DeviceTensor<T, sizeof...(MultiIndex)> out{name, einsums::detail::DEV_ONLY, index...};
    out = create_random_tensor<T, Normalize, MultiIndex...>(name, index...);
    return out;
}
#endif

/**
 * Create a random positive or negative definite matrix.
 * A positive definite matrix is a symmetric matrix whose eigenvalues are all positive.
 * Similarly for negative definite matrices.
 *
 * This function first generates a set of random eigenvectors, making sure they are non-singular.
 * Then, it uses these to form an orthonormal eigenbasis for the new matrix. Then, it generates
 * the eigenvalues. The eigenvalues are distributed using a Maxwell-Boltzmann distribution
 * with the given mean, defaulting to 1. Then, the returned matrix is formed
 * by computing @f$P^TDP@f$. If the mean is negative, then the result will be a negative
 * definite matrix.
 *
 * @param name The name for the matrix.
 * @param rows The number of rows.
 * @param cols The number of columns. Should equal the number of rows.
 * @param mean The mean for the eigenvalues. Defaults to 1.
 * @return A new positive definite or negative definite matrix.
 */
template <typename T = double>
auto create_random_definite(const std::string &name, int rows, int cols, T mean = T{1.0}) -> Tensor<T, 2> {
    if (rows != cols) {
        throw std::runtime_error("Can only make square positive definite matrices.");
    }
    Tensor<T, 2> Evecs = create_random_tensor<T>("name", rows, cols);

    Tensor<T, 2>          Temp = Evecs;
    std::vector<blas_int> pivs;

    // Make sure the eigenvectors are non-singular.
    while (linear_algebra::getrf(&Temp, &pivs) > 0) {
        Evecs = create_random_tensor<T>("name", rows, cols);
        Temp  = Evecs;
    }

    // QR decompose Evecs to get a random matrix of orthonormal eigenvectors.
    auto pair = linear_algebra::qr(Evecs);

    Evecs = linear_algebra::q(std::get<0>(pair), std::get<1>(pair));

    std::default_random_engine engine;

    // Create random eigenvalues. Need to calculate the standard deviation from the mean.
    auto normal = std::normal_distribution<T>(0, std::abs(mean) / T{2.0} / std::numbers::sqrt2_v<T> / std::numbers::inv_sqrtpi_v<T>);

    Tensor<T, 1> Evals("name2", rows);

    for (int i = 0; i < rows; i++) {
        // Maxwell-Boltzmann distribute the eigenvalues. Make sure they are positive.
        do {
            T val1 = normal(engine), val2 = normal(engine), val3 = normal(engine);

            Evals(i) = std::sqrt(val1 * val1 + val2 * val2 + val3 * val3);
            if (mean < T{0.0}) {
                Evals(i) = -Evals(i);
            }
        } while (Evals(i) == T{0.0}); // Make sure they are non-zero.
    }

    // Create the tensor.
    Tensor<T, 2> ret = diagonal(Evals);

    linear_algebra::gemm<false, false>(1.0, ret, Evecs, 0.0, &Temp);
    linear_algebra::gemm<true, false>(1.0, Evecs, Temp, 0.0, &ret);

    ret.set_name(name);

    return ret;
}

/**
 * Create a random positive or negative semi-definite matrix.
 * A positive semi-definite matrix is a symmetric matrix whose eigenvalues are all non-negative.
 * Similarly for negative semi-definite matrices.
 *
 * This function first generates a set of random eigenvectors, making sure they are non-singular.
 * Then, it uses these to form an orthonormal eigenbasis for the new matrix. Then, it generates
 * the eigenvalues. The eigenvalues are distributed using a Maxwell-Boltzmann distribution
 * with the given mean, defaulting to 1. If desired, a number of eigenvalues can be forced to be zero.
 * Then, the returned matrix is formed by computing @f$P^TDP@f$.
 *
 * @param name The name for the matrix.
 * @param rows The number of rows.
 * @param cols The number of columns. Should equal the number of rows.
 * @param mean The mean for the eigenvalues. Defaults to 1. If negative, the result is a negative semi-definite matrix.
 * @param force_zeros The number of elements to force to be zero. Defaults to 1.
 * @return A new positive or negative semi-definite matrix.
 */
template <typename T = double, bool Normalize = false>
auto create_random_semidefinite(const std::string &name, int rows, int cols, T mean = T{1.0}, int force_zeros = 1) -> Tensor<T, 2> {
    if (rows != cols) {
        throw std::runtime_error("Can only make square positive definite matrices.");
    }
    Tensor<T, 2> Evecs = create_random_tensor<T>("name", rows, cols);

    Tensor<T, 2>          Temp = Evecs;
    std::vector<blas_int> pivs;

    // Make sure the eigenvectors are non-singular.
    while (linear_algebra::getrf(&Temp, &pivs) > 0) {
        Evecs = create_random_tensor<T>("name", rows, cols);
        Temp  = Evecs;
    }

    // QR decompose Evecs to get a random matrix of orthonormal eigenvectors.
    auto pair = linear_algebra::qr(Evecs);

    Evecs = linear_algebra::q(std::get<0>(pair), std::get<1>(pair));

    std::default_random_engine engine;

    // Create random eigenvalues. Need to calculate the standard deviation from the mean.
    auto normal = std::normal_distribution<T>(0, std::abs(mean) / T{2.0} / std::numbers::sqrt2_v<T> / std::numbers::inv_sqrtpi_v<T>);

    Tensor<T, 1> Evals("name2", rows);

    for (int i = 0; i < rows; i++) {
        if (i < force_zeros) {
            Evals(i) = T{0.0};
        } else {
            // Maxwell-Boltzmann distribute the eigenvalues. Make sure they are positive.
            T val1 = normal(engine), val2 = normal(engine), val3 = normal(engine);

            Evals(i) = std::sqrt(val1 * val1 + val2 * val2 + val3 * val3);
            if (mean < T{0.0}) {
                Evals(i) = -Evals(i);
            }
        }
    }

    // Create the tensor.
    Tensor<T, 2> ret = diagonal(Evals);

    linear_algebra::gemm<false, false>(1.0, ret, Evecs, 0.0, &Temp);
    linear_algebra::gemm<true, false>(1.0, Evecs, Temp, 0.0, &ret);

    ret.set_name(name);

    return ret;
}

namespace detail {

template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank, typename Tuple, std::size_t... I>
void set_to(TensorType<DataType, Rank> &tensor, DataType value, Tuple const &tuple, std::index_sequence<I...>) {
    tensor(std::get<I>(tuple)...) = value;
}

} // namespace detail

/**
 * @brief Creates a diagonal matrix from a vector.
 *
 * @tparam T The datatype of the underlying data.
 * @param v The input vector.
 * @return A new rank-2 tensor with the diagonal elements set to \p v .
 */
template <typename T>
auto diagonal(const Tensor<T, 1> &v) -> Tensor<T, 2> {
    auto result = create_tensor<T>(v.name(), v.dim(0), v.dim(0));
    zero(result);
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

template <typename T>
auto diagonal_like(const Tensor<T, 1> &v, const Tensor<T, 2> &like) -> Tensor<T, 2> {
    auto result = create_tensor_like(v.name(), like);
    zero(result);
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

/**
 * @brief Create a new tensor with \p name and \p index with ones on the diagonal. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_identity_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_identity_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param name The name of the new tensor.
 * @param index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 */
template <typename T = double, typename... MultiIndex>
auto create_identity_tensor(const std::string &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    static_assert(sizeof...(MultiIndex) >= 1, "Rank parameter doesn't make sense.");

    Tensor<T, sizeof...(MultiIndex)> A{name, std::forward<MultiIndex>(index)...};
    A.zero();

    for (size_t dim = 0; dim < std::get<0>(std::forward_as_tuple(index...)); dim++) {
        detail::set_to(A, T{1.0}, create_tuple<sizeof...(MultiIndex)>(dim), std::make_index_sequence<sizeof...(MultiIndex)>());
    }

    return A;
}

/**
 * @brief Create a new tensor with \p name and \p index filled with ones. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_ones_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_ones_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param name The name of the new tensor.
 * @param index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 */
template <typename T = double, typename... MultiIndex>
auto create_ones_tensor(const std::string &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    static_assert(sizeof...(MultiIndex) >= 1, "Rank parameter doesn't make sense.");

    Tensor<T, sizeof...(MultiIndex)> A{name, std::forward<MultiIndex>(index)...};
    A.set_all(T{1});

    return A;
}

/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 * @code
 * auto a = create_ones_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_tensor_like(a);                  // auto -> Tensor<double, 2>
 * @endcode
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor) -> Tensor<DataType, Rank> {
    return Tensor<DataType, Rank>{tensor.dims()};
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode for the tensor. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> DeviceTensor<DataType, Rank> {
    return DeviceTensor<DataType, Rank>{tensor.dims(), mode};
}
#    endif

/**
 * @brief Creates a new tensor with the same rank, dimensions, and block sizes of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor) -> BlockTensor<DataType, Rank> {
    return BlockTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims()};
}

#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same rank, dimensions, and block sizes of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode for the new tensor. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> BlockDeviceTensor<DataType, Rank> {
    return BlockDeviceTensor<DataType, Rank>{"(unnamed)", mode, tensor.vector_dims()};
}
#    endif
#endif

/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * @code
 * auto a = create_ones_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_tensor_like("b", a);             // auto -> Tensor<double, 2>
 * @endcode
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor) -> Tensor<DataType, Rank> {
    auto result = Tensor<DataType, Rank>{tensor.dims()};
    result.set_name(name);
    return result;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * @code
 * auto a = create_ones_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_tensor_like("b", a);             // auto -> Tensor<double, 2>
 * @endcode
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> DeviceTensor<DataType, Rank> {
    auto result = DeviceTensor<DataType, Rank>{tensor.dims(), mode};
    result.set_name(name);
    return result;
}
#    endif

/**
 * @brief Creates a new tensor with the same, rank, dimensions, and block parameters of the provided tensor.
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor) -> BlockTensor<DataType, Rank> {
    auto result = BlockTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims()};
    result.set_name(name);
    return result;
}

#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same, rank, dimensions, and block parameters of the provided tensor.
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode for the blocks. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> BlockDeviceTensor<DataType, Rank> {
    auto result = BlockDeviceTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims(), mode};
    result.set_name(name);
    return result;
}
#    endif
#endif

/**
 * @brief Creates a new rank-1 tensor filled with digits from \p start to \p stop in \p step increments.
 *
 * @code
 * // auto -> Tensor<double, 1> with data ranging from 0.0 to 9.0
 * auto a = arange<double>(0, 10);
 * @endcode
 *
 * @tparam T Underlying datatype of the tensor
 * @param start Value to start the tensor with
 * @param stop Value to stop the tensor with
 * @param step Increment value
 * @return new rank-1 tensor filled with digits from \p start to \p stop in \p step increments
 */
template <NotComplex T>
auto arange(T start, T stop, T step = T{1}) -> Tensor<T, 1> {
    assert(stop >= start);

    // Determine the number of elements that will be produced
    int nelem = static_cast<int>((stop - start) / step);

    auto result = create_tensor<T>("arange created tensor", nelem);
    zero(result);

    int index{0};
    for (T value = start; value < stop; value += step) {
        result(index++) = value;
    }

    return result;
}

/**
 * @brief Creates a new rank-1 tensor filled with digits from 0 to \p stop .
 *
 * @code
 * // auto -> Tensor<double, 1> with data ranging from 0.0 to 9.0
 * auto a = arange<double>(10);
 * @endcode
 *
 * @tparam T Underlying datatype of the tensor
 * @param stop Value to stop the tensor with
 * @return new rank-1 tensor filled with digits from 0 to \p stop
 */
template <NotComplex T>
auto arange(T stop) -> Tensor<T, 1> {
    return arange(T{0}, stop);
}

struct DisableOMPNestedScope {
    DisableOMPNestedScope() {
        _old_nested = omp_get_nested();
        omp_set_nested(0);
    }

    ~DisableOMPNestedScope() { omp_set_nested(_old_nested); }

  private:
    int _old_nested;
};

struct DisableOMPThreads {
    DisableOMPThreads() {
        _old_max_threads = omp_get_max_threads();
        omp_set_num_threads(1);
    }

    ~DisableOMPThreads() { omp_set_num_threads(_old_max_threads); }

  private:
    int _old_max_threads;
};

#ifdef EINSUMS_USE_CATCH2

/**
 * @struct WithinStrictMatcher
 *
 * Catch2 matcher that matches the strictest range for floating point operations.
 */
template <typename T>
struct WithinStrictMatcher : public Catch::Matchers::MatcherGenericBase {};

template <>
struct WithinStrictMatcher<float> : public Catch::Matchers::MatcherGenericBase {
  private:
    float _value, _scale;

  public:
    WithinStrictMatcher(float value, float scale) : _value(value), _scale(scale) {}

    bool match(float other) const {
        // Minimum error is 5.96e-8, according to LAPACK docs.
        if (_value == 0.0f) {
            return std::abs(other) <= 5.960464477539063e-08f * _scale;
        } else {
            return std::abs((other - _value) / _value) <= 5.960464477539063e-08f * _scale;
        }
    }

    std::string describe() const override {
        return "is within a fraction of " + Catch::StringMaker<float>::convert(5.960464477539063e-08f * _scale) + " to " +
               Catch::StringMaker<float>::convert(_value);
    }

    float get_error() const { return 5.960464477539063e-08f * _scale; }
};

template <>
struct WithinStrictMatcher<double> : public Catch::Matchers::MatcherGenericBase {
  private:
    double _value, _scale;

  public:
    WithinStrictMatcher(double value, double scale) : _value(value), _scale(scale) {}

    bool match(double other) const {
        // Minimum error is 1.1e-16, according to LAPACK docs.
        if (_value == 0.0f) {
            return std::abs(other) <= 1.1102230246251565e-16 * _scale;
        } else {
            return std::abs((other - _value) / _value) <= 1.1102230246251565e-16 * _scale;
        }
    }

    std::string describe() const override {
        return "is within a fraction of " + Catch::StringMaker<double>::convert(1.1102230246251565e-16 * _scale) + " to " +
               Catch::StringMaker<double>::convert(_value);
    }

    double get_error() const { return 1.1102230246251565e-16 * _scale; }
};

template <typename T>
auto WithinStrict(T value, T scale = T{1.0}) -> WithinStrictMatcher<T> {
    return WithinStrictMatcher<T>{value, scale};
}
#endif

} // namespace einsums