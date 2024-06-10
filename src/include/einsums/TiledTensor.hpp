#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"

#include <array>
#include <map>
#include <vector>

namespace einsums {

/**
 * @struct TiledTensorBase
 *
 * Represents a tiled tensor.
 *
 * @tparam TensorType The underlying storage type.
 * @tparam T The type of data being stored.
 * @tparam Rank The tensor rank.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
class TiledTensorBase : detail::TensorBase<T, Rank> {

  protected:
    std::array<std::vector<int>, Rank> _tile_offsets, _tile_sizes;

    std::map<std::array<int, Rank>, TensorType<T, Rank>> _tiles;

    Dim<Rank> _dims;

    size_t _size;

    std::string _name{"(unnamed)"};

  public:
    /**
     * Create a new empty tiled tensor.
     */
    TiledTensorBase() : _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{} {}

    /**
     * Create a new empty tiled tensor with the given grid. If only one grid is given, the grid is applied to all dimensions.
     * Otherwise, the number of grids must match the rank.
     *
     * @param name The name of the tensor.
     * @param sizes The grids to apply.
     */
    template <typename... Sizes>
    TiledTensorBase(std::string name, Sizes... sizes) : _name(name) {
        static_assert(sizeof...(Sizes) == Rank || sizeof...(Sizes) == 1);

        _size = 1;
        if constexpr (sizeof...(Sizes) == Rank) {
            _tile_sizes = std::array<std::vector<int>, Rank>{sizes...};
        } else {
            for (int i = 0; i < Rank; i++) {
                _tile_sizes[i] = std::vector<int>(std::get<0>(sizes...));
            }
        }
        for (int i = 0; i < Rank; i++) {
            _tile_offsets[i] = std::vector<int>();
            int sum          = 0;
            for (int j = 0; j < Rank; j++) {
                _tile_offsets[i].push_back(sum);
                sum += _tile_sizes[i][j];
            }
            _dims[i] = sum;
            _size *= sum;
        }
    }

    /**
     * Copy a tiled tensor.
     *
     * @param other The tensor to be copied.
     */
    TiledTensorBase(const TiledTensorBase<TensorType, T, Rank> &other)
        : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size) {
        for (auto pair : other._tiles) {
            _tiles[pair.first] = TensorType<T, Rank>(pair.second);
        }
    }

    virtual ~TiledTensorBase() = default;

    /**
     * Returns the tile with given coordinates. If the tile is not filled, it will be created.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    TensorType<T, Rank> &tile(MultiIndex... index) {
        std::array<int, Rank> arr_index{index...};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        if (!has_tile(arr_index)) {
            Dim<Rank> dims{};

            for (int i = 0; i < Rank; i++) {
                dims[i] = _tile_sizes[i][arr_index[i]];
            }
            auto new_tile = TensorType<T, Rank>(dims);
            new_tile.zero();
            _tiles[arr_index] = new_tile;
        }

        return _tiles[arr_index];
    }

    /**
     * Returns the tile with given coordinates. If the tile is not filled, this will throw an error.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    const TensorType<T, Rank> &tile(MultiIndex... index) const {
        std::array<int, Rank> arr_index{index...};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        return _tiles.at(arr_index);
    }

    /**
     * Returns the tile with given coordinates. If the tile is not filled, it will be created.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    TensorType<T, Rank> &tile(Storage index) {
        std::array<int, Rank> arr_index{index};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        if (!has_tile(arr_index)) {
            Dim<Rank> dims{};

            for (int i = 0; i < Rank; i++) {
                dims[i] = _tile_sizes[i][arr_index[i]];
            }
            auto new_tile = TensorType<T, Rank>(dims);
            new_tile.zero();
            _tiles[arr_index] = new_tile;
        }

        return _tiles[arr_index];
    }

    /**
     * Returns the tile with given coordinates. If the tile is not filled, this will throw an error.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    const TensorType<T, Rank> &tile(Storage index) const {
        std::array<int, Rank> arr_index{index};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        return _tiles.at(arr_index);
    }

    /**
     * Returns whether a tile exists at a given position, and if it is filled.
     *
     * @param index The position to check for a tile.
     * @return True if there is a tile and it is initialized at this position. False if there is no tile or it is not initialized.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    bool has_tile(MultiIndex... index) const {
        std::array<int, Rank> arr_index{index...};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            if (arr_index[i] >= _tile_sizes[i].size() || arr_index[i] < 0) {
                return false;
            }
        }

        return _tiles.count(arr_index) > 0;
    }

    /**
     * Returns the tile coordinates of a given tensor index.
     *
     * @param index The tensor index to check.
     * @return The coordinates of the tile.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    std::array<int, Rank> tile_of(MultiIndex... index) const {
        std::array<int, Rank> arr_index{index...};
        std::array<int, Rank> out{0};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _dims[i];
            }

            if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
                throw(std::out_of_range("Index not in the tensor!"));
            }

            for (int j = 0; j < _tile_offsets[i].size(); j++) {
                if (arr_index[i] < _tile_offsets[i][j]) {
                    out[i] = j;
                    break;
                }
            }
        }
        return out;
    }

    /**
     * Returns whether a tile exists at a given position, and if it is filled.
     *
     * @param index The position to check for a tile.
     * @return True if there is a tile and it is initialized at this position. False if there is no tile or it is not initialized.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    bool has_tile(Storage index) const {
        std::array<int, Rank> arr_index{index};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            if (arr_index[i] >= _tile_sizes[i].size() || arr_index[i] < 0) {
                return false;
            }
        }

        return _tiles.count(arr_index) > 0;
    }

    /**
     * Returns the tile coordinates of a given tensor index.
     *
     * @param index The tensor index to check.
     * @return The coordinates of the tile.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    std::array<int, Rank> tile_of(Storage index) const {
        std::array<int, Rank> arr_index{index};
        std::array<int, Rank> out{0};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _dims[i];
            }

            if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
                throw(std::out_of_range("Index not in the tensor!"));
            }

            for (int j = 0; j < _tile_offsets[i].size(); j++) {
                if (arr_index[i] < _tile_offsets[i][j]) {
                    out[i] = j;
                    break;
                }
            }
        }
        return out;
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    T operator()(MultiIndex... index) const {
        auto coords = tile_of(index...);

        auto array_ind = std::array<int, Rank>{index...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i][coords[i]];
        }

        if (has_tile(coords)) {
            return std::apply(tile(coords), array_ind);
        } else {
            return T{0};
        }
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    T &operator()(MultiIndex... index) {
        auto coords = tile_of(index...);

        auto array_ind = std::array<int, Rank>{index...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i][coords[i]];
        }

        return std::apply(tile(coords), array_ind);
    }

    /**
     * Sets all entries in the tensor to zero. This clears all tiles. There will be no more tiles after this.
     */
    void zero() { _tiles.clear(); }

    /**
     * Sets all entries in the tensor to zero. This keeps all tiles, just calls zero on the tensors.
     */
    void zero_no_clear() {
        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            tile.second.zero();
        }
    }

    /**
     * Sets all entries to the given value. Initializes all tiles, unless zero is given.
     * If zero is passed, calls @ref zero
     *
     * @param value The value to broadcast.
     */
    void set_all(T value) {
        if (value == T{0}) {
            zero();
            return;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles[tile_index].set_all(value);
        }
    }

    /**
     * Sets all entries to the given value. If a tile does not exist, it is ignored.
     *
     * @param value The value to broadcast.
     */
    void set_all_existing(T value) {
        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            tile.second.set_all(value);
        }
    }

    TiledTensorBase<TensorType, T, Rank> &operator=(T value) {
        set_all(value);
        return *this;
    }

    TiledTensorBase<TensorType, T, Rank> &operator+=(T value) {
        if (value == T{0}) {
            return *this;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles[tile_index] += value;
        }
    }

    TiledTensorBase<TensorType, T, Rank> &operator-=(T value) {
        if (value == T{0}) {
            return *this;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles[tile_index] -= value;
        }
    }

    TiledTensorBase<TensorType, T, Rank> &operator*=(T value) {
        if(value == T{0}) {
            zero();
            return *this;
        }
        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            tile.second *= value;
        }
    }

    TiledTensorBase<TensorType, T, Rank> &operator/=(T value) {
        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            tile.second /= value;
        }
    }
};

template <typename T, size_t Rank>
class TiledTensor : public TiledTensorBase<Tensor, T, Rank> {
  public:
    TiledTensor() = default;

    template <typename... Sizes>
    TiledTensor(std::string name, Sizes... sizes) : TiledTensorBase<Tensor, T, Rank>(name, sizes...) {}

    TiledTensor(const TiledTensor<T, Rank> &other) = default;
};

} // namespace einsums