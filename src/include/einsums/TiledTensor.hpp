#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"

#include <string>

#ifdef __HIP__
#    include "einsums/DeviceTensor.hpp"
#endif

#include <array>
#include <cmath>
#include <map>
#include <vector>

namespace einsums {
namespace detail {
template <size_t Rank>
struct ArrayCompare {
  public:
    bool operator()(const std::array<int, Rank> &x, const std::array<int, Rank> &y) const {
        for (int i = 0; i < Rank; i++) {
            if (x[i] < y[i]) {
                return true;
            }
            if (x[i] > y[i]) {
                return false;
            }
        }
        return false;
    }
    typedef std::array<int, Rank> first_argument_type;
    typedef std::array<int, Rank> second_argument_type;
    typedef bool                  result_type;
};
} // namespace detail

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
class TiledTensorBase : public detail::TensorBase<T, Rank> {

  protected:
    std::array<std::vector<int>, Rank> _tile_offsets, _tile_sizes;

    std::map<std::array<int, Rank>, TensorType<T, Rank>, detail::ArrayCompare<Rank>> _tiles;

    Dim<Rank> _dims;

    size_t _size, _grid_size;

    std::string _name{"(unnamed)"};

    /**
     * Add a tile using the underlying type's preferred method. Also gives it a name.
     */
    virtual void add_tile(std::array<int, Rank> pos) {
        std::string tile_name = _name + " - (";
        Dim<Rank>   dims{};

        for (int i = 0; i < Rank; i++) {
            tile_name += std::to_string(pos[i]);
            dims[i] = _tile_sizes[i][pos[i]];
            if (i != Rank - 1) {
                tile_name += ", ";
            }
        }
        tile_name += ")";

        _tiles.emplace(pos, dims);
        _tiles[pos].set_name(tile_name);
        _tiles[pos].zero();
    }

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
    TiledTensorBase(std::string name, Sizes... sizes) : _name(name), _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{} {
        static_assert(sizeof...(Sizes) == Rank || sizeof...(Sizes) == 1);

        _size = 1;
        if constexpr (sizeof...(Sizes) == Rank) {
            _tile_sizes = std::array<std::vector<int>, Rank>{std::vector<int>(sizes.begin(), sizes.end())...};
        } else {
            for (int i = 0; i < Rank; i++) {
                _tile_sizes[i] = std::vector<int>(sizes.begin()..., sizes.end()...);
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

        _grid_size = 1;

        for (int i = 0; i < Rank; i++) {
            _grid_size *= _tile_offsets[i].size();
        }
    }

    /**
     * Copy a tiled tensor.
     *
     * @param other The tensor to be copied.
     */
    TiledTensorBase(const TiledTensorBase<TensorType, T, Rank> &other)
        : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
          _dims{other._dims} {
        for (auto pair : other._tiles) {
            _tiles[pair.first] = TensorType<T, Rank>(pair.second);
        }
    }

    ~TiledTensorBase() = default;

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

            add_tile(arr_index);
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
            add_tile(arr_index);
        }

        return _tiles[arr_index];
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

            if (arr_index[i] >= _tile_offsets[i][_tile_offsets[i].size() - 1]) {
                out[i] = _tile_offsets[i].size() - 1;
            } else {
                for (int j = 0; j < _tile_offsets[i].size() - 1; j++) {
                    if (arr_index[i] < _tile_offsets[i][j + 1] && arr_index[i] >= _tile_offsets[i][j]) {
                        out[i] = j;
                        break;
                    }
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

            if (arr_index[i] >= _tile_offsets[i][_tile_offsets[i].size() - 1]) {
                out[i] = _tile_offsets[i].size() - 1;
            } else {
                for (int j = 0; j < _tile_offsets[i].size() - 1; j++) {
                    if (arr_index[i] < _tile_offsets[i][j + 1] && arr_index[i] >= _tile_offsets[i][j]) {
                        out[i] = j;
                        break;
                    }
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
            return T{0.0};
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
        auto &out = tile(coords);

        return std::apply(out, array_ind);
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

    TiledTensorBase<TensorType, T, Rank> &operator=(const TiledTensorBase<TensorType, T, Rank> &copy) {
        zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : copy._tiles) {
            add_tile(tile.first);
            _tiles[tile.first] = tile.second;
        }

        return *this;
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
        return *this;
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
        return *this;
    }

    TiledTensorBase<TensorType, T, Rank> &operator*=(T value) {
        if (value == T{0}) {
            zero();
            return *this;
        }
        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            tile.second *= value;
        }
        return *this;
    }

    TiledTensorBase<TensorType, T, Rank> &operator/=(T value) {
        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            tile.second /= value;
        }
        return *this;
    }

    TiledTensorBase<TensorType, T, Rank> &operator+=(const TiledTensorBase<TensorType, T, Rank> &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        // EINSUMS_OMP_PARALLEL_FOR
        for (const auto &tile : other._tiles) {
            if (has_tile(tile.first)) {
                _tiles[tile.first] += tile.second;
            } else {
                _tiles[tile.first] = TensorType<T, Rank>(tile.second);
            }
        }

        return *this;
    }

    TiledTensorBase<TensorType, T, Rank> &operator-=(const TiledTensorBase<TensorType, T, Rank> &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (auto &tile : other._tiles) {
            if (has_tile(tile.first)) {
                _tiles[tile.first] -= tile.second;
            } else {
                _tiles[tile.first] = TensorType<T, Rank>(tile.second);
                _tiles[tile.first] *= -1;
            }
        }

        return *this;
    }

    TiledTensorBase<TensorType, T, Rank> &operator*=(const TiledTensorBase<TensorType, T, Rank> &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            if (other.has_tile(tile.first)) {
                tile.second *= other._tiles[tile.first];
            } else {
                _tiles.erase(tile.first);
            }
        }

        return *this;
    }

    TiledTensorBase<TensorType, T, Rank> &operator/=(const TiledTensorBase<TensorType, T, Rank> &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (auto tile : _tiles) {
            if (other.has_tile(tile.first)) {
                tile.second /= other._tiles[tile.first];
            } else {
                tile.second /= T{0};
            }
        }

        return *this;
    }

    /**
     * Returns the tile offsets.
     */
    std::array<std::vector<int>, Rank> tile_offsets() const { return _tile_offsets; }

    /**
     * Returns the tile offsets along a given dimension.
     *
     * @param i The axis to retrieve.
     *
     */
    std::vector<int> tile_offset(int i = 0) const { return _tile_offsets[i]; }

    /**
     * Returns the tile sizes.
     */
    std::array<std::vector<int>, Rank> tile_sizes() const { return _tile_sizes; }

    /**
     * Returns the tile sizes along a given dimension.
     *
     * @param i The axis to retrieve.
     *
     */
    std::vector<int> tile_size(int i = 0) const { return _tile_sizes[i]; }

    /**
     * Get a reference to the tile map.
     */
    const std::map<std::array<int, Rank>, TensorType<T, Rank>> &tiles() const { return _tiles; }

    /**
     * Get a reference to the tile map.
     */
    std::map<std::array<int, Rank>, TensorType<T, Rank>> &tiles() { return _tiles; }

    /**
     * Get the name.
     */
    std::string name() const { return _name; }

    /**
     * Sets the name.
     *
     * @param val The new name.
     */
    void set_name(std::string val) { _name = val; }

    /**
     * Gets the size of the tensor.
     */
    size_t size() const { return _size; }

    size_t dim(int d) const override { return _dims[d]; }

    /**
     * Gets the number of possible tiles, empty and filled.
     */
    size_t grid_size() const { return _grid_size; }

    /**
     * Gets the number of filled tiles.
     */
    size_t num_filled() const { return _tiles.size(); }

    bool full_view_of_underlying() const { return true; }
};

template <typename T, size_t Rank>
class TiledTensor final : public TiledTensorBase<Tensor, T, Rank> {
  public:
    TiledTensor() = default;

    template <typename... Sizes>
    TiledTensor(std::string name, Sizes... sizes) : TiledTensorBase<Tensor, T, Rank>(name, sizes...) {}

    TiledTensor(const TiledTensor<T, Rank> &other) : TiledTensorBase<Tensor, T, Rank>(other) {}

    ~TiledTensor() = default;
};

template <typename T, size_t Rank>
class TiledTensorView final : public TiledTensorBase<TensorView, T, Rank> {
  private:
    bool _full_view_of_underlying{false};

    void add_tile(std::array<int, Rank> pos) override { throw std::runtime_error("Can't add a tile to a TiledTensorView!"); }

  public:
    TiledTensorView() = default;

    template <typename... Sizes>
    TiledTensorView(std::string name, Sizes... sizes) : TiledTensorBase<TensorView, T, Rank>(name, sizes...) {}

    TiledTensorView(const TiledTensorView<T, Rank> &other) = default;

    ~TiledTensorView() = default;

    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    void insert_tile(std::array<int, Rank> pos, TensorView<T, Rank> &view) { this->_tiles[pos] = view; }
};

#ifdef __HIP__
template <typename T, size_t Rank>
class TiledDeviceTensor final : public TiledTensorBase<DeviceTensor, T, Rank> {
  private:
    detail::HostDeviceMode _mode{detail::DEV_ONLY};

    void add_tile(std::array<int, Rank> pos) override {
        std::string tile_name = _name + " - (";
        Dim<Rank>   dims{};

        for (int i = 0; i < Rank; i++) {
            tile_name += std::to_string(pos[i]);
            dims[i] = _tile_sizes[i][pos[i]];
            if (i != Rank - 1) {
                tile_name += ", ";
            }
        }
        tile_name += ")";

        _tiles.emplace(pos, _mode, dims);
        _tiles[pos].set_name(tile_name);
        _tiles[pos].zero();
    }

  public:
    TiledDeviceTensor() = default;

    template <typename... Sizes>
        requires !(std::is_same_v<Sizes, detail::HostDeviceMode> || ...)
                 TiledDeviceTensor(std::string name, Sizes... sizes, detail::HostDeviceMode mode = detail::DEV_ONLY)
        : _mode{mode},
    TiledTensorBase<DeviceTensor, T, Rank>(name, sizes...) {}

    TiledDeviceTensor(const TiledDeviceTensor<T, Rank> &other) = default;

    ~TiledDeviceTensor() = default;

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
};

template <typename T, size_t Rank>
class TiledDeviceTensorView final : public TiledTensorBase<DeviceTensorView, T, Rank> {
  private:
    bool                   _full_view_of_underlying{false};
    detail::HostDeviceMode _mode{detail::DEV_ONLY};

    void add_tile(std::array<int, Rank> pos) override { throw std::runtime_error("Can't add a tile to a TiledDeviceTensorView!"); }

  public:
    TiledDeviceTensorView() = default;

    template <typename... Sizes>
        requires !(std::is_same_v<Sizes, detail::HostDeviceMode> || ...)
                 TiledDeviceTensorView(std::string name, Sizes... sizes, detail::HostDeviceMode mode = detail::DEV_ONLY)
        : _mode{mode},
    TiledTensorBase<DeviceTensorView, T, Rank>(name, sizes...) {}

    TiledDeviceTensorView(const TiledDeviceTensorView<T, Rank> &other) = default;

    ~TiledDeviceTensorView() = default;

    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    void insert_tile(std::array<int, Rank> pos, DeviceTensorView<T, Rank> &view) { this->_tiles[pos] = view; }
};

#endif

} // namespace einsums

template <size_t Rank, typename T>
void println(const einsums::TiledTensor<T, Rank> &A, TensorPrintOptions options = {}) {
    println("Name: {}", A.name());
    {
        print::Indent const indent{};
        println("Tiled Tensor");
        println("Data Type: {}", type_name<T>());

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= A.tile_offset(i).size();
        }

        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % A.tile_offset(j).size();
                remaining /= A.tile_offset(j).size();
            }

            if (A.has_tile(tile_index)) {
                println(A.tile(tile_index));
            }
        }
    }
}

template <size_t Rank, typename T>
void fprintln(FILE *fp, const einsums::TiledTensor<T, Rank> &A, TensorPrintOptions options = {}) {
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(fp, "Tiled Tensor");
        fprintln(fp, "Data Type: {}", type_name<T>());

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= A.tile_offset(i).size();
        }

        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % A.tile_offset(j).size();
                remaining /= A.tile_offset(j).size();
            }

            if (A.has_tile(tile_index)) {
                fprintln(fp, A.tile(tile_index));
            }
        }
    }
}

template <size_t Rank, typename T>
void fprintln(std::ostream &os, const einsums::TiledTensor<T, Rank> &A, TensorPrintOptions options = {}) {
    fprintln(os, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(os, "Tiled Tensor");
        fprintln(os, "Data Type: {}", type_name<T>());

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= A.tile_offset(i).size();
        }

        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % A.tile_offset(j).size();
                remaining /= A.tile_offset(j).size();
            }

            if (A.has_tile(tile_index)) {
                println(A.tile(tile_index));
            }
        }
    }
}