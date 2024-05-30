//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#ifdef __HIP__
#include "einsums/DeviceTensor.hpp"
#endif
#include "einsums/OpenMP.h"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/ComplexTraits.hpp"

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

    {
        static std::chrono::high_resolution_clock::time_point const beginning = std::chrono::high_resolution_clock::now();

        // std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now() - beginning;

        // re.seed(d.count());
    }

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
    auto result = create_tensor(v.name(), v.dim(0), v.dim(0));
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
    requires requires {
        requires CoreRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires !BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const TensorType<DataType, Rank> &tensor) -> Tensor<DataType, Rank> {
    return Tensor<DataType, Rank>{tensor.dims()};
}

#ifdef __HIP__
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
    requires requires {
        requires DeviceRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires !BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> DeviceTensor<DataType, Rank> {
    return DeviceTensor<DataType, Rank>{tensor.dims(), mode};
}
#endif

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
    requires requires {
        requires CoreRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const TensorType<DataType, Rank> &tensor) -> BlockTensor<DataType, Rank> {
    return BlockTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims()};
}

#ifdef __HIP__
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
    requires requires {
        requires DeviceRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> BlockDeviceTensor<DataType, Rank> {
    return BlockDeviceTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims(), mode};
}
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
    requires requires {
        requires CoreRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires !BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor) -> Tensor<DataType, Rank> {
    auto result = Tensor<DataType, Rank>{tensor.dims()};
    result.set_name(name);
    return result;
}

#ifdef __HIP__
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
    requires requires {
        requires DeviceRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires !BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> DeviceTensor<DataType, Rank> {
    auto result = DeviceTensor<DataType, Rank>{tensor.dims(), mode};
    result.set_name(name);
    return result;
}
#endif

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
    requires requires {
        requires CoreRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor) -> BlockTensor<DataType, Rank> {
    auto result = BlockTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims()};
    result.set_name(name);
    return result;
}

#ifdef __HIP__
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
    requires requires {
        requires DeviceRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
        requires BlockRankTensor<TensorType<DataType, Rank>, Rank, DataType>;
    }
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> BlockDeviceTensor<DataType, Rank> {
    auto result = BlockDeviceTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims(), mode};
    result.set_name(name);
    return result;
}
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

template <typename T>
auto divmod(T n, T d) -> std::tuple<T, T> {
    return {n / d, n % d};
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

} // namespace einsums