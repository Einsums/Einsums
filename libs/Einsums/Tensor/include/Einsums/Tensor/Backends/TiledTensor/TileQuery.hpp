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
TensorType &TiledTensor<T, TensorType, KeyType>::tile(MultiIndex... index) {
    KeyType const key = convert_coords_to_key(index...);

    if (!has_tile(key)) {
        add_tile(key);
    }

    return _tiles.at(key);
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
TensorType const &TiledTensor<T, TensorType, KeyType>::tile(MultiIndex... index) const {
    KeyType const key = convert_coords_to_key(index...);

    return _tiles.at(key);
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer Storage>
TensorType &TiledTensor<T, TensorType, KeyType>::tile(Storage const &index) {
    KeyType const key = convert_coords_to_key(index);

    if (!has_tile(key)) {
        add_tile(key);
    }

    return _tiles.at(key);
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer Storage>
TensorType const &TiledTensor<T, TensorType, KeyType>::tile(Storage const &index) const {
    KeyType const key = convert_coords_to_key(index);

    return _tiles.at(key);
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
bool TiledTensor<T, TensorType, KeyType>::has_tile(MultiIndex... index) const {
    try {
        KeyType const key = convert_coords_to_key(index...);

        return _tiles.count(key) > 0;
    } catch (std::out_of_range &e) {
        return false;
    } catch (num_argument_error &e) {
        EINSUMS_LOG_INFO("Wrong number of arguments passed to has_tile. Returning false.");
        return false;
    }
    // Other errors mean something else went wrong, and don't indicate whether we have a certain tile.

    // We need to return here just in case of fall through (it won't happen).
    return false;
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
KeyType TiledTensor<T, TensorType, KeyType>::tile_of(MultiIndex... index) const {
    if (sizeof...(MultiIndex) != _rank) {
        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                "Wrong number of arguments passed to tile request function! Expected number should match the rank.");
    }

    KeyType out;

    auto index_tuple = std::make_tuple(index...);

    if constexpr (key_has_resize) {
        out.resize(_rank);
    }

    for_sequence<sizeof...(MultiIndex)>([&](auto n) {
        size_t const cast_n = static_cast<size_t>(n);
        auto         idx    = std::get<cast_n>(index_tuple);

        if constexpr (std::is_signed_v<std::remove_cvref_t<decltype(idx)>>) {
            if (idx < 0) {
                idx += _dims[cast_n];
            }
        }

        if (idx >= _dims[cast_n] || idx < 0) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index ({})was not within [0, {}) after adjustment.", print::ordinal(cast_n),
                                    idx, _dims[cast_n]);
        }

        if (idx >= *_tile_offsets[cast_n].rbegin()) {
            out[cast_n] = static_cast<typename KeyType::value_type>(_tile_offsets[cast_n].size() - 1);
        } else {
            for (size_t j = 0; j < _tile_offsets[cast_n].size() - 1; j++) {
                if (idx < _tile_offsets[cast_n].at(j + 1) && idx >= _tile_offsets[cast_n].at(j)) {
                    out[cast_n] = static_cast<typename KeyType::value_type>(j);
                    break;
                }
            }
        }
    });

    return out;
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer Storage>
bool TiledTensor<T, TensorType, KeyType>::has_tile(Storage const &index) const {
    try {
        KeyType const key = convert_coords_to_key(index);

        return _tiles.count(key) > 0;
    } catch (std::out_of_range &e) {
        return false;
    } catch (num_argument_error &e) {
        EINSUMS_LOG_INFO("Wrong number of arguments passed to has_tile. Returning false.");
        return false;
    }
    // Other errors mean something else went wrong, and don't indicate whether we have a certain tile.

    // We need to return here just in case of fall through (it won't happen).
    return false;
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer Storage>
KeyType TiledTensor<T, TensorType, KeyType>::tile_of(Storage const &index) const {
    KeyType out;

    if constexpr (key_has_resize) {
        out.resize(_rank);
    }

    auto index_iter = index.begin();

    for (int i = 0; i < _rank && index_iter != index.end(); i++, ++index_iter) {
        auto idx = *index_iter;

        if constexpr (std::is_signed_v<typename Storage::value_type>) {
            if (idx < 0) {
                idx += _dims[i];
            }
        }

        if (idx < 0 || idx >= _dims[i]) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index ({}) is not within [0, {}) after adjustment.", print::ordinal(i), idx,
                                    _dims[i]);
        }

        if (idx >= *_tile_offsets[i].rbegin()) {
            out[i] = _tile_offsets[i].size() - 1;
        } else {
            for (int j = 0; j < _tile_offsets[i].size() - 1; j++) {
                if (idx < _tile_offsets[i].at(j + 1) && idx >= _tile_offsets[i].at(j)) {
                    out[i] = j;
                    break;
                }
            }
        }
    }
    return out;
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... Index>
bool TiledTensor<T, TensorType, KeyType>::has_zero_size(Index... index) const {
    KeyType const key = convert_coords_to_key(index...);

    for (int i = 0; i < _rank; i++) {
        if (_tile_sizes[i].at(key[i]) == 0) {
            return true;
        }
    }

    return false;
}

template <typename T, typename TensorType, typename KeyType>
template <ContainerOrInitializer Storage>
bool TiledTensor<T, TensorType, KeyType>::has_zero_size(Storage const &index) const {
    KeyType const key = convert_coords_to_key(index);

    for (int i = 0; i < _rank; i++) {
        if (_tile_sizes[i].at(key[i]) == 0) {
            return true;
        }
    }

    return false;
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
KeyType TiledTensor<T, TensorType, KeyType>::convert_coords_to_key(MultiIndex... index) const {
    if (sizeof...(MultiIndex) != _rank) {
        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                "Wrong number of arguments passed to tile request function! Expected number should match the rank.");
    }
    KeyType out;

    if constexpr (key_has_resize) {
        out.resize(_rank);
    }

    auto index_tuple = std::make_tuple(index...);

    for_sequence<sizeof...(MultiIndex)>([&](auto n) {
        auto idx = std::get<static_cast<size_t>(n)>(index_tuple);

        if constexpr (std::is_signed_v<std::remove_cvref_t<decltype(idx)>>) {
            if (idx < 0) {
                idx += _tile_sizes[static_cast<size_t>(n)].size();
            }
        }

        if (idx >= _tile_sizes[static_cast<size_t>(n)].size() || idx < 0) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index ({})was not within [0, {}) after adjustment.",
                                    print::ordinal(static_cast<size_t>(n)), idx, _tile_sizes[static_cast<size_t>(n)].size());
        }

        out[static_cast<size_t>(n)] = static_cast<typename KeyType::value_type>(idx);
    });

    return out;
}

template <typename T, typename TensorType, typename KeyType>
template <Container Index>
KeyType TiledTensor<T, TensorType, KeyType>::convert_coords_to_key(Index const &index) const {
    if (index.size() != _rank) {
        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                "Wrong number of elements passed to tile request function! Expected number should match the rank.");
    }
    KeyType out;

    if constexpr (key_has_resize) {
        out.resize(_rank);
    }

    auto in_iter         = index.begin();
    auto key_iter        = out.begin();
    auto tile_sizes_iter = _tile_sizes.begin();

    for (; key_iter != out.end() && tile_sizes_iter != _tile_sizes.end() && in_iter != index.end();
         ++key_iter, ++tile_sizes_iter, ++in_iter) {
        typename Index::value_type idx = *in_iter;
        if constexpr (std::is_signed_v<typename Index::value_type>) {
            if (idx < 0) {
                idx += tile_sizes_iter->size();
            }
        }

        *key_iter = idx;

        if (idx >= tile_sizes_iter->size() || *key_iter < 0) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index ({})was not within [0, {}) after adjustment.",
                                    print::ordinal(std::distance(out.begin(), key_iter)), *key_iter, tile_sizes_iter->size());
        }
    }
    return out;
}

template <typename T, typename TensorType, typename KeyType>
template <std::integral... MultiIndex>
KeyType TiledTensor<T, TensorType, KeyType>::convert_index_to_key(MultiIndex... index) const {
    if (sizeof...(MultiIndex) != _rank) {
        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                "Wrong number of arguments passed to tile request function! Expected number should match the rank.");
    }
    KeyType out;

    if constexpr (key_has_resize) {
        out.resize(_rank);
    }

    auto index_tuple = std::make_tuple(index...);

    for_sequence<sizeof...(MultiIndex)>([&out, &index_tuple, this](auto n) {
        auto idx = std::get<static_cast<size_t>(n)>(index_tuple);

        if constexpr (std::is_signed_v<std::remove_cvref_t<decltype(idx)>>) {
            if (idx < 0) {
                idx += _dims[static_cast<size_t>(n)];
            }
        }

        if (idx >= _dims[static_cast<size_t>(n)] || idx < 0) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index ({})was not within [0, {}) after adjustment.",
                                    print::ordinal(static_cast<size_t>(n)), idx, _dims[static_cast<size_t>(n)]);
        }

        out[static_cast<size_t>(n)] = static_cast<typename KeyType::value_type>(idx);
    });

    return out;
}

template <typename T, typename TensorType, typename KeyType>
template <Container Index>
KeyType TiledTensor<T, TensorType, KeyType>::convert_index_to_key(Index const &index) const {
    if (index.size() != _rank) {
        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                "Wrong number of elements passed to tile request function! Expected number should match the rank.");
    }
    KeyType out;

    if constexpr (key_has_resize) {
        out.resize(_rank);
    }

    auto in_iter   = index.begin();
    auto key_iter  = out.begin();
    auto dims_iter = _dims.cbegin();

    for (; key_iter != out.end() && dims_iter != _dims.cend() && in_iter != index.end(); ++key_iter, ++dims_iter, ++in_iter) {
        auto idx = *in_iter;
        if constexpr (std::is_signed_v<typename Index::value_type>) {
            if (idx < 0) {
                idx += *dims_iter;
            }
        }

        *key_iter = idx;

        if (*key_iter >= *dims_iter || *key_iter < 0) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index ({})was not within [0, {}) after adjustment.",
                                    print::ordinal(std::distance(out.begin(), key_iter)), *key_iter, *dims_iter);
        }
    }
    return out;
}

} // namespace einsums::tensor_base
