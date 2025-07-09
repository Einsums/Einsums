//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <bit>
#include <complex>
#include <concepts>
#include <limits>
#include <numbers>
#include <omp.h>
#include <random>
#include <string>

namespace einsums {

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
 * @param distribution The random distribution to use for generating the random numbers.
 * @param index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 */
template <typename T = double, bool Normalize = false, typename Distribution, std::integral... MultiIndex>
    requires requires(Distribution dist) {
        { dist(einsums::random_engine) } -> std::same_as<T>;
    }
auto create_random_tensor(std::string const &name, Distribution &&distribution, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating random tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(name, std::forward<MultiIndex>(index)...);
    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < A.size(); i++) {
        A.data()[i] = distribution(einsums::random_engine);
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

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations. The data generated will
 * be between -1 and 1 for reals.
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
template <typename T = double, bool Normalize = false, std::integral... MultiIndex>
auto create_random_tensor(std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    if constexpr (IsComplexV<T>) {
        return create_random_tensor<T, Normalize>(name, detail::unit_circle_distribution<T>(), index...);
    } else {
        return create_random_tensor<T, Normalize>(name, std::uniform_real_distribution<T>(-1, 1), index...);
    }
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
 * @param distribution The random distribution to use for generating the random numbers.
 * @param index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 */
template <typename T = double, bool Normalize = false, typename Distribution, Container Indices>
auto create_random_tensor(std::string const &name, Distribution &&dist, Indices const &indices) -> RuntimeTensor<T> {
    EINSUMS_LOG_TRACE("creating random runtime tensor {}, {}", name, indices);

    RuntimeTensor<T> A(name, indices);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < A.size(); i++) {
        A.data()[i] = dist(einsums::random_engine);
    }

    if constexpr (Normalize) {
        if (indices.size() == 2) {
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
    }

    return A;
}

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations. The data generated will
 * be between -1 and 1 for reals.
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
template <typename T = double, bool Normalize = false, Container Indices>
auto create_random_tensor(std::string const &name, Indices const &index) -> RuntimeTensor<T> {
    if constexpr (IsComplexV<T>) {
        return create_random_tensor<T, Normalize>(name, detail::unit_circle_distribution<T>(), index);
    } else {
        return create_random_tensor<T, Normalize>(name, std::uniform_real_distribution<T>(-1, 1), index);
    }
}

#if defined(EINSUMS_COMPUTE_CODE)
template <typename T = double, bool Normalize = false, typename... MultiIndex>
auto create_random_gpu_tensor(std::string const &name, MultiIndex... index) -> DeviceTensor<T, sizeof...(MultiIndex)> {
    DeviceTensor<T, sizeof...(MultiIndex)> out{name, einsums::detail::DEV_ONLY, index...};
    out = create_random_tensor<T, Normalize, MultiIndex...>(name, index...);
    return out;
}
#endif

} // namespace einsums
