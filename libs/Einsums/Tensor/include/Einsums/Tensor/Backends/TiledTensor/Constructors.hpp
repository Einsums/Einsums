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
// template <typename T, typename TensorType, typename KeyType>
// TiledTensor<T, TensorType, KeyType>::

namespace einsums::tensor_base {

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer... Sizes>
    requires(!ContainerOrInitializer<typename Sizes::value_type> && ... && true)
TiledTensor<T, TensorType, KeyType>::TiledTensor(std::string const &name, size_t rank, Sizes const &...sizes)
    : _name(name), _tile_offsets(rank), _tile_sizes(rank), _tiles(), _size(0), _dims(rank), _rank{rank} {

    _size = 1;
    if (sizeof...(Sizes) == rank) {
        auto size_tuple = std::make_tuple(sizes...);
        for_sequence<sizeof...(Sizes)>([&](auto i) {
            auto &size = std::get<i>(size_tuple);

            this->_tile_sizes[static_cast<size_t>(i)] = std::vector<size_t>(size.size());
            for (size_t j = 0; j < size.size(); j++) {
                this->_tile_sizes[static_cast<size_t>(i)][j] = size[j];
            }
        });
    } else if constexpr (sizeof...(Sizes) == 1) {
        for (size_t i = 0; i < rank; i++) {
            _tile_sizes[i] = std::vector<size_t>(sizes.size()...);

            for (size_t j = 0; j < _tile_sizes[i].size(); j++) {
                _tile_sizes[i][j] = (sizes[j] + ...); // Expand the parameter pack to make the compiler happy.
            }
        }
    } else {
        EINSUMS_THROW_EXCEPTION(num_argument_error, "The wrong number of arguments was passed to constructor! Must either be one argument "
                                                    "for a square tensor or the same number of arguments as the rank.");
    }
    for (size_t i = 0; i < rank; i++) {
        _tile_offsets[i] = std::vector<size_t>();
        _tile_offsets[i].reserve(_tile_sizes[i].size());

        size_t sum = 0;
        for (size_t j = 0; j < _tile_sizes[i].size(); j++) {
            _tile_offsets[i].push_back(sum);
            sum += _tile_sizes[i].at(j);
        }
        _dims[i] = sum;
        _size *= sum;
    }

    _grid_size = 1;

    for (size_t i = 0; i < rank; i++) {
        _grid_size *= _tile_offsets[i].size();
    }
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer ContainerType>
    requires(ContainerOrInitializer<typename ContainerType::value_type> &&
             std::is_integral_v<typename ContainerType::value_type::value_type>)
TiledTensor<T, TensorType, KeyType>::TiledTensor(std::string const &name, ContainerType const &sizes)
    : _name(name), _tile_offsets(sizes.size()), _tile_sizes(sizes.size()), _tiles(), _size(0), _dims(sizes.size()), _rank{sizes.size()} {
    for (size_t i = 0; i < _rank; i++) {
        _tile_sizes[i] = std::vector<size_t>(sizes[i].size());

        for (size_t j = 0; j < sizes[i].size(); j++) {
            _tile_sizes[i][j] = sizes[i][j];
        }
    }
    _size = 1;
    for (size_t i = 0; i < _rank; i++) {
        _tile_offsets[i] = std::vector<size_t>();
        _tile_offsets[i].reserve(_tile_sizes[i].size());
        size_t sum = 0;
        for (size_t j = 0; j < _tile_sizes[i].size(); j++) {
            _tile_offsets[i].push_back(sum);
            sum += _tile_sizes[i].at(j);
        }
        _dims[i] = sum;
        _size *= sum;
    }

    _grid_size = 1;

    for (size_t i = 0; i < _rank; i++) {
        _grid_size *= _tile_offsets[i].size();
    }
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType>::TiledTensor(TiledTensor<T, TensorType, KeyType> const &other)
    : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
      _dims{other._dims}, _grid_size{other._grid_size}, _rank{other._rank} {
    for (auto &pair : other._tiles) {
        _tiles.insert_or_assign(pair.first, pair.second);
    }
}

template <typename T, typename TensorType, typename KeyType>
template <TypedTensorConcept<T> OtherTensor, typename OtherKey>
TiledTensor<T, TensorType, KeyType>::TiledTensor(TiledTensor<T, OtherTensor, OtherKey> const &other)
    : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
      _dims{other._dims}, _grid_size{other._grid_size}, _rank{other._rank} {
    for (auto &pair : other._tiles) {
        if constexpr (std::is_same_v<OtherKey, KeyType>) {
            _tiles.insert_or_assign(pair.first, pair.second);
        } else {
            KeyType cast_key;

            if constexpr (key_has_resize) {
                cast_key.resize();
            }

            for (size_t i = 0; i < _rank; i++) {
                cast_key[i] = pair.first[i];
            }

            _tiles.try_emplace(cast_key, pair.second);
        }
    }
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType>::TiledTensor(TiledTensor<T, TensorType, KeyType> &&other)
    : _tile_offsets(std::move(other._tile_offsets)), _tile_sizes(std::move(other._tile_sizes)), _name(std::move(other._name)),
      _size(other._size), _tiles(std::move(other._tiles)), _dims{std::move(other._dims)}, _grid_size{other._grid_size}, _rank{other._rank} {
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator=(TiledTensor<T, TensorType, KeyType> const &copy) {
    zero();
    _tile_sizes   = copy._tile_sizes;
    _tile_offsets = copy._tile_offsets;
    _dims         = copy._dims;
    _name         = copy._name;
    _size         = copy._size;
    _grid_size    = copy._grid_size;
    _rank         = copy._rank;

    for (auto const &tile : copy._tiles) {
        add_tile(tile.first);
        _tiles.at(tile.first) = tile.second;
    }

    return *this;
}

template <typename T, typename TensorType, typename KeyType>
TiledTensor<T, TensorType, KeyType> &TiledTensor<T, TensorType, KeyType>::operator=(TiledTensor<T, TensorType, KeyType> &&copy) {
    _tiles        = std::move(copy._tiles);
    _tile_sizes   = std::move(copy._tile_sizes);
    _tile_offsets = std::move(copy._tile_offsets);
    _dims         = std::move(copy._dims);
    _name         = std::move(copy._name);
    _size         = copy._size;
    _grid_size    = copy._grid_size;
    _rank         = copy._rank;

    return *this;
}

template <typename T, typename TensorType, typename KeyType>
template <typename TOther, TensorConcept TensorOther, typename KeyOther>
TiledTensor<T, TensorType, KeyType> &
TiledTensor<T, TensorType, KeyType>::operator=(TiledTensor<TOther, TensorOther, KeyOther> const &copy) {
    zero();
    _tile_sizes   = copy._tile_sizes;
    _tile_offsets = copy._tile_offsets;
    _dims         = copy._dims;
    _name         = copy._name;
    _size         = copy._size;
    _grid_size    = copy._grid_size;
    _rank         = copy._rank;

    for (auto const &tile : copy._tiles) {

        if constexpr (std::is_same_v<KeyType, KeyOther>) {
            _tiles[tile.first] = tile.second;
        } else {
            KeyType convert = convert_coords_to_key(tile.first);

            _tiles[convert] = tile.second;
        }
    }

    return *this;
}

} // namespace einsums::tensor_base
