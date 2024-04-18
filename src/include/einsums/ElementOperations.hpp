//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"

#include <algorithm>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::element_operations)

BEGIN_EINSUMS_NAMESPACE_HPP(detail)
template <typename vector, typename Functor>
void omp_loop(vector &data, Functor functor) {
    LabeledSection0();

    // TODO: This only works for Tensors not their views because we assume data is a std::vector
    EINSUMS_OMP_PARALLEL {
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

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto sum(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    // TODO: This currently only works with Tensor's not TensorViews. And it needs to be OpenMP'd
    T result = std::accumulate(tensor.vector_data().begin(), tensor.vector_data().end(), T{0});
    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto max(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    // TODO: This currently only works with Tensor's not TensorViews. And it needs to be OpenMP'd
    auto result = std::max_element(tensor.vector_data().begin(), tensor.vector_data().end());
    return *result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto min(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    // TODO: This currently only works with Tensor's not TensorViews. And it needs to be OpenMP'd
    auto result = std::min_element(tensor.vector_data().begin(), tensor.vector_data().end());
    return *result;
}

BEGIN_EINSUMS_NAMESPACE_HPP(new_tensor)

using einsums::element_operations::max; // nolint
using einsums::element_operations::sum; // nolint

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto abs(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    LabeledSection0();

    auto result = create_tensor_like(tensor);
    result      = tensor;

    ::einsums::element_operations::detail::omp_loop(result.vector_data(), [](T &value) { return std::abs(value); });

    return result;
}

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

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto exp(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    LabeledSection0();

    auto result = create_tensor_like(tensor);
    result      = tensor;
    auto &data  = result.vector_data();

    ::einsums::element_operations::detail::omp_loop(data, [&](T &value) { return std::exp(value); });

    return result;
}

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