//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Tensor/TiledTensor.hpp>

namespace einsums::tensor_base {

template <typename T, size_t rank, typename TensorType>
template <ContainerOrInitializer... Sizes>
    requires(!ContainerOrInitializer<typename Sizes::value_type> && ... && true)
TiledTensor<T, rank, TensorType>::TiledTensor(std::string name, Sizes const &...sizes)
    : _name(name), _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{} {
    static_assert(sizeof...(Sizes) == rank || sizeof...(Sizes) == 1);

    _size = 1;
    if constexpr (sizeof...(Sizes) == rank) {
        auto size_tuple = std::make_tuple(sizes...);
        for_sequence<rank>([&](auto i) {
            auto &size = std::get<i>(size_tuple);

            this->_tile_sizes[(int)i] = std::vector<int>(size.size());
            for (int j = 0; j < size.size(); j++) {
                this->_tile_sizes[(int)i][j] = size[j];
            }
        });
    } else {
        for (int i = 0; i < rank; i++) {
            _tile_sizes[i] = std::vector<int>(sizes.size()...);

            for (int j = 0; j < _tile_sizes[i].size(); j++) {
                _tile_sizes[i][j] = (sizes[j] + ...);
            }
        }
    }
    for (int i = 0; i < rank; i++) {
        _tile_offsets[i] = std::vector<int>();
        _tile_offsets[i].reserve(_tile_sizes[i].size());

        int sum = 0;
        for (int j = 0; j < _tile_sizes[i].size(); j++) {
            _tile_offsets[i].push_back(sum);
            sum += _tile_sizes[i].at(j);
        }
        _dims[i] = sum;
        _size *= sum;
    }

    _grid_size = 1;

    for (int i = 0; i < rank; i++) {
        _grid_size *= _tile_offsets[i].size();
    }
}

template <typename T, size_t rank, typename TensorType>
template <Container ContainerType>
    requires(Container<typename ContainerType::value_type> && std::is_integral_v<typename ContainerType::value_type::value_type>)
TiledTensor<T, rank, TensorType>::TiledTensor(std::string name, ContainerType const &sizes)
    : _name(name), _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{} {
    if (sizes.size() != rank) {
        EINSUMS_THROW_EXCEPTION(num_argument_error, "Wrong number of grid sizes passed to TiledTensor constructor!");
    }
    for (int i = 0; i < rank; i++) {
        _tile_sizes[i] = std::vector<int>(sizes[i].size());

        for (int j = 0; j < sizes[i].size(); j++) {
            _tile_sizes[i][j] = sizes[i][j];
        }
    }
    _size = 1;
    for (int i = 0; i < rank; i++) {
        _tile_offsets[i] = std::vector<int>();
        _tile_offsets[i].reserve(_tile_sizes[i].size());
        int sum = 0;
        for (int j = 0; j < _tile_sizes[i].size(); j++) {
            _tile_offsets[i].push_back(sum);
            sum += _tile_sizes[i].at(j);
        }
        _dims[i] = sum;
        _size *= sum;
    }

    _grid_size = 1;

    for (int i = 0; i < rank; i++) {
        _grid_size *= _tile_offsets[i].size();
    }
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType>::TiledTensor(TiledTensor<T, rank, TensorType> const &other)
    : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
      _dims{other._dims}, _grid_size{other._grid_size} {
    for (auto &pair : other._tiles) {
        _tiles.insert_or_assign(pair.first, pair.second);
    }
}

template <typename T, size_t rank, typename TensorType>
template <TRTensorConcept<rank, T> OtherTensor>
TiledTensor<T, rank, TensorType>::TiledTensor(TiledTensor<T, rank, OtherTensor> const &other)
    : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
      _dims{other._dims}, _grid_size{other._grid_size} {
    for (auto &pair : other._tiles) {
        _tiles.insert_or_assign(pair.first, pair.second);
    }
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... MultiIndex>
    requires(sizeof...(MultiIndex) == rank)
TensorType &TiledTensor<T, rank, TensorType>::tile(MultiIndex... index) {
    std::array<int, rank> arr_index{static_cast<int>(index)...};

    for (int i = 0; i < rank; i++) {
        if (arr_index[i] < 0) {
            arr_index[i] += _tile_sizes[i].size();
        }

        assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
    }

    if (!has_tile(arr_index)) {
        Dim<rank> dims{};

        for (int i = 0; i < rank; i++) {
            dims[i] = _tile_sizes[i].at(arr_index[i]);
        }

        add_tile(arr_index);
    }

    return _tiles.at(arr_index);
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... MultiIndex>
    requires(sizeof...(MultiIndex) == rank)
TensorType const &TiledTensor<T, rank, TensorType>::tile(MultiIndex... index) const {
    std::array<int, rank> arr_index{static_cast<int>(index)...};

    for (int i = 0; i < rank; i++) {
        if (arr_index[i] < 0) {
            arr_index[i] += _tile_sizes[i].size();
        }

        assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
    }

    return _tiles.at(arr_index);
}

template <typename T, size_t rank, typename TensorType>
template <typename Storage>
    requires(!std::integral<Storage>)
TensorType const &TiledTensor<T, rank, TensorType>::tile(Storage index) const {
    std::array<int, rank> arr_index;

    for (int i = 0; i < rank; i++) {
        arr_index[i] = static_cast<int>(index[i]);
        if (arr_index[i] < 0) {
            arr_index[i] += _tile_sizes[i].size();
        }

        assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
    }

    return _tiles.at(arr_index);
}

template <typename T, size_t rank, typename TensorType>
template <typename Storage>
    requires(!std::integral<Storage>)
TensorType &TiledTensor<T, rank, TensorType>::tile(Storage index) {
    std::array<int, rank> arr_index{index};

    for (int i = 0; i < rank; i++) {
        if (arr_index[i] < 0) {
            arr_index[i] += _tile_sizes[i].size();
        }

        assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
    }

    if (!has_tile(arr_index)) {
        Dim<rank> dims{};

        for (int i = 0; i < rank; i++) {
            dims[i] = _tile_sizes[i].at(arr_index[i]);
        }
        add_tile(arr_index);
    }

    return _tiles.at(arr_index);
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... MultiIndex>
    requires(sizeof...(MultiIndex) == rank)
bool TiledTensor<T, rank, TensorType>::has_tile(MultiIndex... index) const {
    std::array<int, rank> arr_index{static_cast<int>(index)...};

    for (int i = 0; i < rank; i++) {
        if (arr_index[i] < 0) {
            arr_index[i] += _tile_sizes[i].size();
        }

        if (arr_index[i] >= _tile_sizes[i].size() || arr_index[i] < 0) {
            return false;
        }
    }

    return _tiles.count(arr_index) > 0;
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... MultiIndex>
    requires(sizeof...(MultiIndex) == rank)
std::array<int, rank> TiledTensor<T, rank, TensorType>::tile_of(MultiIndex... index) const {
    std::array<int, rank> arr_index{static_cast<int>(index)...};
    std::array<int, rank> out{0};

    for (int i = 0; i < rank; i++) {
        if (arr_index[i] < 0) {
            arr_index[i] += _dims[i];
        }

        if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index not in the tensor!");
        }

        if (arr_index[i] >= _tile_offsets[i].at(_tile_offsets[i].size() - 1)) {
            out[i] = _tile_offsets[i].size() - 1;
        } else {
            for (int j = 0; j < _tile_offsets[i].size() - 1; j++) {
                if (arr_index[i] < _tile_offsets[i].at(j + 1) && arr_index[i] >= _tile_offsets[i].at(j)) {
                    out[i] = j;
                    break;
                }
            }
        }
    }
    return out;
}

template <typename T, size_t rank, typename TensorType>
template <typename Storage>
    requires(!std::integral<Storage>)
bool TiledTensor<T, rank, TensorType>::has_tile(Storage index) const {
    std::array<int, rank> arr_index;

    for (int i = 0; i < rank; i++) {
        arr_index[i] = static_cast<int>(index[i]);
        if (arr_index[i] < 0) {
            index[i] += _tile_sizes[i].size();
        }

        if (arr_index[i] >= _tile_sizes[i].size() || arr_index[i] < 0) {
            return false;
        }
    }

    return _tiles.count(arr_index) > 0;
}

template <typename T, size_t rank, typename TensorType>
template <typename Storage>
    requires(!std::integral<Storage>)
std::array<int, rank> TiledTensor<T, rank, TensorType>::tile_of(Storage index) const {
    std::array<int, rank> arr_index;
    std::array<int, rank> out{0};

    for (int i = 0; i < rank; i++) {
        arr_index[i] = static_cast<int>(index[i]);
        if (arr_index[i] < 0) {
            arr_index[i] += _dims[i];
        }

        if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index not in the tensor!");
        }

        if (arr_index[i] >= _tile_offsets[i].at(_tile_offsets[i].size() - 1)) {
            out[i] = _tile_offsets[i].size() - 1;
        } else {
            for (int j = 0; j < _tile_offsets[i].size() - 1; j++) {
                if (arr_index[i] < _tile_offsets[i].at(j + 1) && arr_index[i] >= _tile_offsets[i].at(j)) {
                    out[i] = j;
                    break;
                }
            }
        }
    }
    return out;
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... MultiIndex>
    requires(sizeof...(MultiIndex) == rank)
T TiledTensor<T, rank, TensorType>::operator()(MultiIndex... index) const {
    auto coords = tile_of(index...);

    auto array_ind = std::array<int64_t, rank>{static_cast<int64_t>(index)...};

    // Find the index in the tile.
    for (int i = 0; i < rank; i++) {
        if (array_ind[i] < 0) {
            array_ind[i] += _dims[i];
        }
        array_ind[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), array_ind);
    } else {
        return T{0.0};
    }
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... MultiIndex>
    requires(sizeof...(MultiIndex) == rank)
T TiledTensor<T, rank, TensorType>::at(MultiIndex... index) const {
    auto coords = tile_of(index...);

    auto array_ind = std::array<int64_t, rank>{static_cast<int64_t>(index)...};

    // Find the index in the tile.
    for (int i = 0; i < rank; i++) {
        if (array_ind[i] < 0) {
            array_ind[i] += _dims[i];
        }
        array_ind[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), array_ind);
    } else {
        return T{0.0};
    }
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... MultiIndex>
    requires(sizeof...(MultiIndex) == rank)
T &TiledTensor<T, rank, TensorType>::operator()(MultiIndex... index) {
    auto coords = tile_of(index...);

    auto array_ind = std::array<int64_t, rank>{static_cast<int64_t>(index)...};

    // Find the index in the tile.
    for (int i = 0; i < rank; i++) {
        if (array_ind[i] < 0) {
            array_ind[i] += _dims[i];
        }
        array_ind[i] -= _tile_offsets[i].at(coords[i]);
    }
    auto &out = tile(coords);

    return subscript_tensor(out, array_ind);
}

template <typename T, size_t rank, typename TensorType>
template <typename ContainerType>
    requires requires {
        requires !std::is_integral_v<ContainerType>;
        requires !std::is_same_v<ContainerType, Dim<rank>>;
        requires !std::is_same_v<ContainerType, Stride<rank>>;
        requires !std::is_same_v<ContainerType, Offset<rank>>;
        requires !std::is_same_v<ContainerType, Range>;
    }
T TiledTensor<T, rank, TensorType>::operator()(ContainerType const &index) const {
    if (index.size() < rank) {
        [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Not enough indices passed to Tensor!");
    } else if (index.size() > rank) {
        [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Too many indices passed to Tensor!");
    }
    auto coords = tile_of(index);

    std::array<std::int64_t, rank> array_ind;

    for (size_t i = 0; i < rank; i++) {
        array_ind[i] = index[i];
    }

    // Find the index in the tile.
    for (int i = 0; i < rank; i++) {
        if (array_ind[i] < 0) {
            array_ind[i] += _dims[i];
        }
        array_ind[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), array_ind);
    } else {
        return T{0.0};
    }
}

template <typename T, size_t rank, typename TensorType>
template <typename ContainerType>
    requires requires {
        requires !std::is_integral_v<ContainerType>;
        requires !std::is_same_v<ContainerType, Dim<rank>>;
        requires !std::is_same_v<ContainerType, Stride<rank>>;
        requires !std::is_same_v<ContainerType, Offset<rank>>;
        requires !std::is_same_v<ContainerType, Range>;
    }
T TiledTensor<T, rank, TensorType>::at(ContainerType const &index) const {
    if (index.size() < rank) {
        [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Not enough indices passed to Tensor!");
    } else if (index.size() > rank) {
        [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Too many indices passed to Tensor!");
    }
    auto coords = tile_of(index);

    std::array<std::int64_t, rank> array_ind;

    for (size_t i = 0; i < rank; i++) {
        array_ind[i] = index[i];
    }

    // Find the index in the tile.
    for (int i = 0; i < rank; i++) {
        if (array_ind[i] < 0) {
            array_ind[i] += _dims[i];
        }
        array_ind[i] -= _tile_offsets[i].at(coords[i]);
    }

    if (has_tile(coords)) {
        return subscript_tensor(tile(coords), array_ind);
    } else {
        return T{0.0};
    }
}

template <typename T, size_t rank, typename TensorType>
template <typename ContainerType>
    requires requires {
        requires !std::is_integral_v<ContainerType>;
        requires !std::is_same_v<ContainerType, Dim<rank>>;
        requires !std::is_same_v<ContainerType, Stride<rank>>;
        requires !std::is_same_v<ContainerType, Offset<rank>>;
        requires !std::is_same_v<ContainerType, Range>;
    }
T &TiledTensor<T, rank, TensorType>::operator()(ContainerType const &index) {
    if (index.size() < rank) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
    } else if (index.size() > rank) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
    }
    auto coords = tile_of(index);

    std::array<std::int64_t, rank> array_ind;

    for (size_t i = 0; i < rank; i++) {
        array_ind[i] = index[i];
    }

    // Find the index in the tile.
    for (int i = 0; i < rank; i++) {
        if (array_ind[i] < 0) {
            array_ind[i] += _dims[i];
        }
        array_ind[i] -= _tile_offsets[i].at(coords[i]);
    }
    auto &out = tile(coords);

    return subscript_tensor(out, array_ind);
}

template <typename T, size_t rank, typename TensorType>
void TiledTensor<T, rank, TensorType>::zero() {
    _tiles.clear();
}

template <typename T, size_t rank, typename TensorType>
void TiledTensor<T, rank, TensorType>::zero_no_clear() {
    for (auto &tile : _tiles) {
        tile.second.zero();
    }
}

template <typename T, size_t rank, typename TensorType>
void TiledTensor<T, rank, TensorType>::set_all(T value) {
    if (value == T{0}) {
        zero();
        return;
    }

    // Find the number of tiles.
    long num_tiles = 1;
    for (int i = 0; i < rank; i++) {
        num_tiles *= _tile_offsets[i].size();
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (long i = 0; i < num_tiles; i++) {
        std::array<int, rank> tile_index{};
        long                  remaining = i;

        // Turn sentinel into an index.
        for (int j = 0; j < rank; j++) {
            tile_index[j] = remaining % _tile_offsets[j].size();
            remaining /= _tile_offsets[j].size();
        }

        // Set the tile index.
        _tiles.at(tile_index).set_all(value);
    }
}

template <typename T, size_t rank, typename TensorType>
void TiledTensor<T, rank, TensorType>::set_all_existing(T value) {
    for (auto &tile : _tiles) {
        tile.second.set_all(value);
    }
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator=(TiledTensor<T, rank, TensorType> const &copy) {
    zero();
    _tile_sizes   = copy._tile_sizes;
    _tile_offsets = copy._tile_offsets;
    _dims         = copy._dims;
    _name         = copy._name;
    _size         = copy._size;
    _grid_size    = copy._grid_size;

    for (auto const &tile : copy._tiles) {
        add_tile(tile.first);
        _tiles.at(tile.first) = tile.second;
    }

    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator=(TiledTensor<T, rank, TensorType> &&move) {
    _tile_sizes   = std::move(move._tile_sizes);
    _tile_offsets = std::move(move._tile_offsets);
    _dims         = std::move(move._dims);
    _name         = std::move(move._name);
    _size         = std::move(move._size);
    _grid_size    = std::move(move._grid_size);
    _tiles        = std::move(move._tiles);

    return *this;
}

template <typename T, size_t rank, typename TensorType>
template <TiledTensorConcept TensorOther>
    requires(SameUnderlyingAndRank<TiledTensor<T, rank, TensorType>, TensorOther>)
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator=(TensorOther const &copy) {
    zero();
    _tile_sizes   = copy._tile_sizes;
    _tile_offsets = copy._tile_offsets;
    _dims         = copy._dims;
    _name         = copy._name;
    _size         = copy._size;
    _grid_size    = copy._grid_size;

    for (auto const &tile : copy._tiles) {
        add_tile(tile.first);
        _tiles.at(tile.first) = tile.second;
    }

    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator=(T value) {
    set_all(value);
    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator+=(T value) {
    if (value == T{0}) {
        return *this;
    }

    // Find the number of tiles.
    long num_tiles = 1;
    for (int i = 0; i < rank; i++) {
        num_tiles *= _tile_offsets[i].size();
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (long i = 0; i < num_tiles; i++) {
        std::array<int, rank> tile_index{};
        long                  remaining = i;

        // Turn sentinel into an index.
        for (int j = 0; j < rank; j++) {
            tile_index[j] = remaining % _tile_offsets[j].size();
            remaining /= _tile_offsets[j].size();
        }

        // Set the tile index.
        _tiles.at(tile_index) += value;
    }
    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator-=(T value) {
    if (value == T{0}) {
        return *this;
    }

    // Find the number of tiles.
    long num_tiles = 1;
    for (int i = 0; i < rank; i++) {
        num_tiles *= _tile_offsets[i].size();
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (long i = 0; i < num_tiles; i++) {
        std::array<int, rank> tile_index{};
        long                  remaining = i;

        // Turn sentinel into an index.
        for (int j = 0; j < rank; j++) {
            tile_index[j] = remaining % _tile_offsets[j].size();
            remaining /= _tile_offsets[j].size();
        }

        // Set the tile index.
        _tiles.at(tile_index) -= value;
    }
    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator*=(T value) {
    if (value == T{0}) {
        zero();
        return *this;
    }
    for (auto &tile : _tiles) {
        tile.second *= value;
    }
    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator/=(T value) {
    for (auto &tile : _tiles) {
        tile.second /= value;
    }
    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator+=(TiledTensor<T, rank, TensorType> const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto const &tile : other._tiles) {
        if (has_tile(tile.first)) {
            _tiles.at(tile.first) += tile.second;
        } else {
            add_tile(tile.first);
            _tiles.at(tile.first) = TensorType(tile.second);
        }
    }

    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator-=(TiledTensor<T, rank, TensorType> const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto const &tile : other._tiles) {
        if (has_tile(tile.first)) {
            _tiles.at(tile.first) -= tile.second;
        } else {
            add_tile(tile.first);
            _tiles.at(tile.first) = TensorType(tile.second);
            _tiles.at(tile.first) *= -1;
        }
    }

    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator*=(TiledTensor<T, rank, TensorType> const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto const &tile : _tiles) {
        if (other.has_tile(tile.first)) {
            tile.second *= other._tiles.at(tile.first);
        } else {
            _tiles.erase(tile.first);
        }
    }

    return *this;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType> &TiledTensor<T, rank, TensorType>::operator/=(TiledTensor const &other) {
    if (_tile_sizes != other._tile_sizes) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
    }

    for (auto const &tile : _tiles) {
        if (other.has_tile(tile.first)) {
            tile.second /= other._tiles.at(tile.first);
        } else {
            tile.second /= T{0};
        }
    }

    return *this;
}

template <typename T, size_t rank, typename TensorType>
std::array<std::vector<int>, rank> TiledTensor<T, rank, TensorType>::tile_offsets() const {
    return _tile_offsets;
}

template <typename T, size_t rank, typename TensorType>
std::vector<int> TiledTensor<T, rank, TensorType>::tile_offset(int i) const {
    return _tile_offsets.at(i);
}

template <typename T, size_t rank, typename TensorType>
std::array<std::vector<int>, rank> TiledTensor<T, rank, TensorType>::tile_sizes() const {
    return _tile_sizes;
}

template <typename T, size_t rank, typename TensorType>
std::vector<int> TiledTensor<T, rank, TensorType>::tile_size(int i) const {
    return _tile_sizes.at(i);
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType>::map_type const &TiledTensor<T, rank, TensorType>::tiles() const {
    return _tiles;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType>::map_type &TiledTensor<T, rank, TensorType>::tiles() {
    return _tiles;
}

template <typename T, size_t rank, typename TensorType>
std::string const &TiledTensor<T, rank, TensorType>::name() const {
    return _name;
}

template <typename T, size_t rank, typename TensorType>
void TiledTensor<T, rank, TensorType>::set_name(std::string const &val) {
    _name = val;
}

template <typename T, size_t rank, typename TensorType>
size_t TiledTensor<T, rank, TensorType>::size() const {
    return _size;
}

template <typename T, size_t rank, typename TensorType>
size_t TiledTensor<T, rank, TensorType>::grid_size() const {
    return _grid_size;
}

template <typename T, size_t rank, typename TensorType>
size_t TiledTensor<T, rank, TensorType>::grid_size(int i) const {
    return _tile_sizes[i].size();
}

template <typename T, size_t rank, typename TensorType>
size_t TiledTensor<T, rank, TensorType>::num_filled() const {
    return _tiles.size();
}

template <typename T, size_t rank, typename TensorType>
bool TiledTensor<T, rank, TensorType>::full_view_of_underlying() const {
    return true;
}

template <typename T, size_t rank, typename TensorType>
size_t TiledTensor<T, rank, TensorType>::dim(int d) const {

    int new_d = d;

    if (new_d < 0) {
        new_d += rank;
    }

    return _dims.at(d);
}

template <typename T, size_t rank, typename TensorType>
Dim<rank> TiledTensor<T, rank, TensorType>::dims() const {
    return _dims;
}

template <typename T, size_t rank, typename TensorType>
template <std::integral... Index>
    requires(sizeof...(Index) == rank)
bool TiledTensor<T, rank, TensorType>::has_zero_size(Index... index) const {
    std::array<int, rank> arr_index{static_cast<int>(index)...};

    for (int i = 0; i < rank; i++) {
        if (_tile_sizes[i].at(arr_index[i]) == 0) {
            return true;
        }
    }

    return false;
}

template <typename T, size_t rank, typename TensorType>
template <typename Storage>
    requires(!std::integral<Storage>)
bool TiledTensor<T, rank, TensorType>::has_zero_size(Storage const &index) const {
    for (int i = 0; i < rank; i++) {
        if (_tile_sizes[i].at(index[i]) == 0) {
            return true;
        }
    }

    return false;
}

template <typename T, size_t rank, typename TensorType>
TiledTensor<T, rank, TensorType>::operator TensorType() const {
    TensorType out(_dims);
    out.set_name(name());

    Stride<rank> tile_strides;

    size_t tiles = 1;

    for (ptrdiff_t i = rank - 1; i >= 0; i--) {
        tile_strides[i] = tiles;
        tiles *= grid_size(i);
    }

    for (size_t tile = 0; tile < tiles; tile++) {
        std::array<int64_t, rank> tile_index;

        sentinel_to_indices(tile, tile_strides, tile_index);

        if (!this->has_tile(tile_index) || this->has_zero_size(tile_index)) {
            continue;
        } else {
            // Calculate the view ranges.
            thread_local std::array<Range, rank> ranges;

            for (size_t i = 0; i < rank; i++) {
                ranges[i] =
                    Range{this->tile_offset(i)[tile_index[i]], this->tile_offset(i)[tile_index[i]] + this->tile_size(i)[tile_index[i]]};
            }

            // Create the view.
            auto tile_view = std::apply(out, ranges);

            // Assign.
            tile_view = this->tile(tile_index);
        }
    }

    return out;
}

} // namespace einsums::tensor_base
