//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>

#include <cstddef>
#include <tuple>

namespace einsums::detail {

template <TensorConcept TensorType, typename DataType, typename Tuple, std::size_t... I>
void set_to(TensorType &tensor, DataType value, Tuple const &tuple, std::index_sequence<I...>) {
    tensor(std::get<I>(tuple)...) = (typename TensorType::ValueType)value;
}

} // namespace einsums::detail