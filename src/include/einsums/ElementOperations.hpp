/*
 * Copyright (c) 2022 Justin Turney
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file ElementOperations.hpp
 * 
 * Contains functions for performing element operations.
 */

#pragma once

#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include <algorithm>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::element_operations)

BEGIN_EINSUMS_NAMESPACE_HPP(detail)

/**
 * @todo Omp-a Loop-a doopity doo. What in the world could this do?
 * 
 * @param data Vector of data.
 * @param functor A functor???
 */
template <typename vector, typename Functor>
void omp_loop(vector &data, Functor functor) {
    LabeledSection0();

    // TODO: This only works for Tensors not their views because we assume data is a std::vector
#pragma omp parallel
    {
        auto tid       = omp_get_thread_num();
        auto chunksize = data.size() / omp_get_num_threads();
        auto begin     = data.begin() + chunksize * tid;
        auto end       = (tid == omp_get_num_threads() - 1) ? data.end() : begin + chunksize;

        EINSUMS_OMP_SIMD
        for (auto i = begin; i < end; i++) {
            *i = functor(*i);
        }
    }
}
END_EINSUMS_NAMESPACE_HPP(detail)

/**
 * Computes a sum.
 * @todo Needs more information.
 *
 * @param tensor A tensor to sum.
 *
 * @return A sum???
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto sum(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    // TODO: This currently only works with Tensor's not TensorViews. And it needs to be OpenMP'd
    T result = std::accumulate(tensor.vector_data().begin(), tensor.vector_data().end(), T{0});
    return result;
}

/**
 * Finds the max element in the tensor.
 *
 * @param tensor The tensor to evaulate.
 *
 * @return The max element in the tensor.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto max(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    // TODO: This currently only works with Tensor's not TensorViews. And it needs to be OpenMP'd
    auto result = std::max_element(tensor.vector_data().begin(), tensor.vector_data().end());
    return *result;
}

BEGIN_EINSUMS_NAMESPACE_HPP(new_tensor)

using einsums::element_operations::max; // nolint
using einsums::element_operations::sum; // nolint

/**
 * Compute the absolute value of each element in the tensor.
 *
 * @param tensor The tensor to absolute value.
 * 
 * @return A new tensor whose elements are the absolute value of the input.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto abs(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    LabeledSection0();

    auto result = create_tensor_like(tensor);
    result      = tensor;

    ::einsums::element_operations::detail::omp_loop(result.vector_data(), [](T &value) { return std::abs(value); });

    return result;
}

/**
 * Finds the reciprocal of each element in the tensor.
 *
 * @param tensor The tensor to invert.
 *
 * @return A tensor whose elements are the reciprocal of the input's.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto invert(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    LabeledSection0();

    auto result = create_tensor_like(tensor);
    result      = tensor;
    auto &data  = result.vector_data();

    // TODO: This only works for Tensor's not their views.
    ::einsums::element_operations::detail::omp_loop(data, [&](T &value) { return T{1} / value; });

    return result;
}

/**
 * Exponentiate the elements of the input tensor.
 *
 * @param tensor The tensor to exponentiate.
 *
 * @return A tensor whose elements are the exponetial of the input's.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto exp(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    LabeledSection0();

    auto result = create_tensor_like(tensor);
    result      = tensor;
    auto &data  = result.vector_data();

    ::einsums::element_operations::detail::omp_loop(data, [&](T &value) { return std::exp(value); });

    return result;
}

/**
 * Perform scalar multiplication on a tensor.
 *
 * @param scale The value to scale by.
 * @param tensor The tensor to scale.
 *
 * @return A new tensor whose elements have been scaled.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto scale(const T &scale, const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    LabeledSection0();

    auto result = create_tensor_like(tensor);
    result      = tensor;
    auto &data  = result.vector_data();

    ::einsums::element_operations::detail::omp_loop(data, [&](T &value) { return scale * value; });

    return result;
}

END_EINSUMS_NAMESPACE_HPP(new_tensor)
END_EINSUMS_NAMESPACE_HPP(einsums::element_operations)
