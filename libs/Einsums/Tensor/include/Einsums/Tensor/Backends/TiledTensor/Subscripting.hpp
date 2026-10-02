//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorBase/HashFunctions.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// For copy-paste:
/*
template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType>::
*/

namespace einsums::tensor_base {

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
T TiledTensor<T, TensorType, KeyType>::operator()(MultiIndex... index) const {
    KeyType full_key = convert_index_to_key(index...);

    KeyType const coords = tile_of(index...);

    for (size_t i = 0; i < _rank; i++) {
        full_key[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), full_key);
    } else {
        return T{0.0};
    }
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
T TiledTensor<T, TensorType, KeyType>::at(MultiIndex... index) const {
    KeyType full_key = convert_index_to_key(index...);

    KeyType const coords = tile_of(index...);

    for (size_t i = 0; i < _rank; i++) {
        full_key[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), full_key);
    } else {
        return T{0.0};
    }
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
T &TiledTensor<T, TensorType, KeyType>::operator()(MultiIndex... index) {
    KeyType full_key = convert_index_to_key(index...);

    KeyType const coords = tile_of(index...);

    for (size_t i = 0; i < _rank; i++) {
        full_key[i] -= _tile_offsets[i].at(coords[i]);
    }

    auto &out = tile(coords);

    return subscript_tensor(out, full_key);
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer ContainerType>
T TiledTensor<T, TensorType, KeyType>::operator()(ContainerType const &index) const {
    KeyType full_key = convert_index_to_key(index);

    KeyType const coords = tile_of(index);

    for (size_t i = 0; i < _rank; i++) {
        full_key[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), full_key);
    } else {
        return T{0.0};
    }
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer ContainerType>
T TiledTensor<T, TensorType, KeyType>::at(ContainerType const &index) const {
    KeyType full_key = convert_index_to_key(index);

    KeyType const coords = tile_of(index);

    for (size_t i = 0; i < _rank; i++) {
        full_key[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), full_key);
    } else {
        return T{0.0};
    }
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer ContainerType>
T &TiledTensor<T, TensorType, KeyType>::operator()(ContainerType const &index) {
    KeyType full_key = convert_index_to_key(index);

    KeyType const coords = tile_of(index);

    for (size_t i = 0; i < _rank; i++) {
        full_key[i] -= _tile_offsets[i].at(coords[i]);
    }

    auto &out = tile(coords);

    return subscript_tensor(out, full_key);
}

} // namespace einsums::tensor_base
