//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>

#include <type_traits>

namespace einsums {

/**
 * @brief Subscripts into a tensor, selecting between the subscript method or the function call syntax.
 */
template <RankTensorConcept TensorType, typename... MultiArgs>
    requires requires {
        requires std::remove_cvref_t<TensorType>::Rank == sizeof...(MultiArgs);
        requires(std::is_integral_v<std::remove_cvref_t<MultiArgs>> && ...);
        requires FastSubscriptableConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, MultiArgs &&...args) -> decltype(tensor.subscript(std::forward<MultiArgs>(args)...)) {
    return tensor.subscript(std::forward<MultiArgs>(args)...);
}
#ifndef DOXYGEN
template <RankTensorConcept TensorType, typename... MultiArgs>
    requires requires {
        requires std::remove_cvref_t<TensorType>::Rank == sizeof...(MultiArgs);
        requires(std::is_integral_v<std::remove_cvref_t<MultiArgs>> && ...);
        requires !FastSubscriptableConcept<TensorType>;
        requires FunctionTensorConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, MultiArgs &&...args) -> decltype(tensor(std::forward<MultiArgs>(args)...)) {
    return tensor(std::forward<MultiArgs>(args)...);
}

template <RankTensorConcept TensorType, typename ContainerType>
    requires requires {
        requires !std::is_integral_v<ContainerType>;
        requires FastSubscriptableConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, ContainerType const &args) -> decltype(tensor.subscript(args)) {
    return tensor.subscript(args);
}

template <RankTensorConcept TensorType, typename ContainerType>
    requires requires {
        requires !std::is_integral_v<ContainerType>;
        requires !FastSubscriptableConcept<TensorType>;
        requires FunctionTensorConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, ContainerType const &args) -> decltype(tensor(args)) {
    return tensor(args);
}
#endif

} // namespace einsums