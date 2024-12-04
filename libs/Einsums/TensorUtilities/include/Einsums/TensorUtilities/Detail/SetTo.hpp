//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <tuple>

namespace einsums::detail {

template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank, typename Tuple, std::size_t... I>
void set_to(TensorType<DataType, Rank> &tensor, DataType value, Tuple const &tuple, std::index_sequence<I...>) {
    tensor(std::get<I>(tuple)...) = value;
}

} // namespace einsums::detail