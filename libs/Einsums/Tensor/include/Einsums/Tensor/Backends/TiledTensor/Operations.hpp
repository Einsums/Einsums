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
void TiledTensor<T, TensorType, KeyType>::zero() {
    _tiles.clear();
}

template <typename T, typename TensorType, typename KeyType>
void TiledTensor<T, TensorType, KeyType>::zero_no_clear() {
    for (auto &tile : _tiles) {
        tile.second.zero();
    }
}

template <typename T, typename TensorType, typename KeyType>
void TiledTensor<T, TensorType, KeyType>::set_all(T value) {
    if (value == T{0.0}) {
        zero();
        return;
    }

    // Find the number of tiles.
    size_t num_tiles = 1;
    for (size_t i = 0; i < _rank; i++) {
        num_tiles *= _tile_offsets[i].size();
    }

    KeyType tile_index;

    if constexpr (key_has_resize) {
        tile_index.resize(_rank);
    }

    for (size_t i = 0; i < num_tiles; i++) {
        size_t remaining = i;

        // Turn sentinel into an index.
        for (int j = 0; j < _rank; j++) {
            tile_index[j] = remaining % _tile_offsets[j].size();
            remaining /= _tile_offsets[j].size();
        }

        // Set the tile index.
        _tiles[tile_index].set_all(value);
    }
}

template <typename T, typename TensorType, typename KeyType>
void TiledTensor<T, TensorType, KeyType>::set_all_existing(T value) {
    for (auto &tile : _tiles) {
        tile.second.set_all(value);
    }
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator=(T value) {
    set_all(value);
    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator+=(T value) {
    if (value == T{0.0}) {
        return *this;
    }

    // Find the number of tiles.
    size_t num_tiles = 1;
    for (size_t i = 0; i < _rank; i++) {
        num_tiles *= _tile_offsets[i].size();
    }

    // Set up the keys for each thread.
    std::vector<KeyType> keys(omp_get_max_threads());

    if constexpr (key_has_resize) {
        for (auto &key : keys) {
            key.resize(_rank);
        }
    }

    // Enumerate over all possible tiles, even the ones that don't exist.
    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < num_tiles; i++) {
        KeyType &tile_index = keys[omp_get_thread_num()];
        size_t   remaining  = i;

        // Turn sentinel into an index.
        for (int j = 0; j < _rank; j++) {
            tile_index[j] = remaining % _tile_offsets[j].size();
            remaining /= _tile_offsets[j].size();
        }

        // Set the tile index.
        _tiles.at(tile_index) += value;
    }
    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator-=(T value) {
    if (value == T{0.0}) {
        return *this;
    }

    // Find the number of tiles.
    size_t num_tiles = 1;
    for (size_t i = 0; i < _rank; i++) {
        num_tiles *= _tile_offsets[i].size();
    }

    // Set up the keys for each thread.
    std::vector<KeyType> keys(omp_get_max_threads());

    if constexpr (key_has_resize) {
        for (auto &key : keys) {
            key.resize(_rank);
        }
    }

    // Enumerate over all possible tiles, even the ones that don't exist.
    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < num_tiles; i++) {
        KeyType &tile_index = keys[omp_get_thread_num()];
        size_t   remaining  = i;

        // Turn sentinel into an index.
        for (int j = 0; j < _rank; j++) {
            tile_index[j] = remaining % _tile_offsets[j].size();
            remaining /= _tile_offsets[j].size();
        }

        // Set the tile index.
        _tiles.at(tile_index) -= value;
    }
    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator*=(T value) {
    if (value == T{0.0}) {
        zero();
        return *this;
    }
    for (auto &tile : _tiles) {
        tile.second *= value;
    }
    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator/=(T value) {
    for (auto &tile : _tiles) {
        tile.second /= value;
    }
    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator+=(TiledTensor<T, TensorType, KeyType> const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto &tile : other._tiles) {
        if (has_tile(tile.first)) {
            _tiles.at(tile.first) += tile.second;
        } else {
            add_tile(tile.first);
            _tiles.at(tile.first) = TensorType(tile.second);
        }
    }

    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator-=(TiledTensor<T, TensorType, KeyType> const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto &tile : other._tiles) {
        if (has_tile(tile.first)) {
            _tiles.at(tile.first) -= tile.second;
        } else {
            add_tile(tile.first);
            _tiles.at(tile.first) = TensorType(tile.second);
            _tiles.at(tile.first) *= T{-1.0};
        }
    }

    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator*=(TiledTensor<T, TensorType, KeyType> const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto &tile : _tiles) {
        if (other.has_tile(tile.first)) {
            tile.second *= other._tiles.at(tile.first);
        } else {
            _tiles.erase(tile.first);
        }
    }

    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator/=(TiledTensor<T, TensorType, KeyType> const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto &tile : _tiles) {
        if (other.has_tile(tile.first)) {
            tile.second /= other._tiles.at(tile.first);
        } else {
            tile.second /= T{0.0};
        }
    }

    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType>::operator TensorType() const {
    TensorType out(_dims);
    out.set_name(name());

    out.zero();

    std::vector<size_t> tile_strides(_rank);

    size_t tiles = 1;

    for (ptrdiff_t i = _rank - 1; i >= 0; i--) {
        tile_strides[i] = tiles;
        tiles *= grid_size(i);
    }

    for (size_t tile = 0; tile < tiles; tile++) {
        KeyType tile_index;

        if constexpr (key_has_resize) {
            tile_index.resize(_rank);
        }

        sentinel_to_indices(tile, tile_strides, tile_index);

        if (!this->has_tile(tile_index) || this->has_zero_size(tile_index)) {
            continue;
        } else {
            // Calculate the view ranges.
            std::vector<Range> ranges(_rank);

            for (size_t i = 0; i < _rank; i++) {
                ranges[i] =
                    Range{this->tile_offset(i)[tile_index[i]], this->tile_offset(i)[tile_index[i]] + this->tile_size(i)[tile_index[i]]};
            }

            // Create the view.
            auto tile_view = out(ranges);

            // Assign.
            tile_view = this->tile(tile_index);
        }
    }

    return out;
}

} // namespace einsums::tensor_base
