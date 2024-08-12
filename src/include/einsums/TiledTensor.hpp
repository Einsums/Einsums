#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/utility/HashFuncs.hpp"
#include "einsums/utility/TensorBases.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <string>

#ifdef __HIP__
#    include "einsums/DeviceTensor.hpp"
#endif

#include <array>
#include <cmath>
#include <concepts>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace einsums {

namespace tensor_props {

/**
 * @struct TiledTensorBase
 *
 * Represents a tiled tensor. Is a lockable type.
 *
 * @tparam TensorType The underlying storage type.
 * @tparam T The type of data being stored.
 * @tparam Rank The tensor rank.
 */
template <typename T, size_t Rank, typename TensorType>
struct TiledTensorBase : public virtual CollectedTensorBase<T, Rank, TensorType>,
                         virtual TiledTensorBaseNoExtra,
                         virtual LockableTensorBase,
                         virtual AlgebraOptimizedTensor {
  public:
    using map_type = typename std::unordered_map<std::array<int, Rank>, TensorType, einsums::hashes::container_hash<std::array<int, Rank>>>;

  protected:
    std::array<std::vector<int>, Rank> _tile_offsets, _tile_sizes;

    map_type _tiles;

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
            for (int j = 0; j < _tile_sizes[i].size(); j++) {
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
    TiledTensorBase(const TiledTensorBase<T, Rank, TensorType> &other)
        : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
          _dims{other._dims} {
        for (auto pair : other._tiles) {
            _tiles[pair.first] = TensorType(pair.second);
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
    TensorType &tile(MultiIndex... index) {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

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
    const TensorType &tile(MultiIndex... index) const {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

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
    const TensorType &tile(Storage index) const {
        std::array<int, Rank> arr_index;

        for (int i = 0; i < Rank; i++) {
            arr_index[i] = static_cast<int>(index[i]);
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
    TensorType &tile(Storage index) {
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
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

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
        std::array<int, Rank> arr_index{static_cast<int>(index)...};
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
        std::array<int, Rank> arr_index;

        for (int i = 0; i < Rank; i++) {
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

    /**
     * Returns the tile coordinates of a given tensor index.
     *
     * @param index The tensor index to check.
     * @return The coordinates of the tile.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    std::array<int, Rank> tile_of(Storage index) const {
        std::array<int, Rank> arr_index;
        std::array<int, Rank> out{0};

        for (int i = 0; i < Rank; i++) {
            arr_index[i] = static_cast<int>(index[i]);
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

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

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

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

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
        for (auto &tile : _tiles) {
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
        for (auto &tile : _tiles) {
            tile.second.set_all(value);
        }
    }

    TiledTensorBase &operator=(const TiledTensorBase &copy) {
        zero();
        _tile_sizes   = copy._tile_sizes;
        _tile_offsets = copy._tile_offsets;
        _dims         = copy._dims;
        _name         = copy._name;
        _size         = copy._size;
        _grid_size    = copy._grid_size;

        for (const auto &tile : copy._tiles) {
            add_tile(tile.first);
            _tiles[tile.first] = tile.second;
        }

        return *this;
    }

    TiledTensorBase &operator=(T value) {
        set_all(value);
        return *this;
    }

    TiledTensorBase &operator+=(T value) {
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

    TiledTensorBase &operator-=(T value) {
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

    TiledTensorBase &operator*=(T value) {
        if (value == T{0}) {
            zero();
            return *this;
        }
        for (auto &tile : _tiles) {
            tile.second *= value;
        }
        return *this;
    }

    TiledTensorBase &operator/=(T value) {
        for (auto &tile : _tiles) {
            tile.second /= value;
        }
        return *this;
    }

    TiledTensorBase &operator+=(const TiledTensorBase &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : other._tiles) {
            if (has_tile(tile.first)) {
                _tiles[tile.first] += tile.second;
            } else {
                _tiles[tile.first] = TensorType(tile.second);
            }
        }

        return *this;
    }

    TiledTensorBase &operator-=(const TiledTensorBase &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : other._tiles) {
            if (has_tile(tile.first)) {
                _tiles[tile.first] -= tile.second;
            } else {
                _tiles[tile.first] = TensorType(tile.second);
                _tiles[tile.first] *= -1;
            }
        }

        return *this;
    }

    TiledTensorBase &operator*=(const TiledTensorBase &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : _tiles) {
            if (other.has_tile(tile.first)) {
                tile.second *= other._tiles[tile.first];
            } else {
                _tiles.erase(tile.first);
            }
        }

        return *this;
    }

    TiledTensorBase &operator/=(const TiledTensorBase &other) {
        if (_tile_sizes != other._tile_sizes) {
            throw std::runtime_error("Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : _tiles) {
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
    std::vector<int> tile_offset(int i = 0) const { return _tile_offsets.at(i); }

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
    std::vector<int> tile_size(int i = 0) const { return _tile_sizes.at(i); }

    /**
     * Get a reference to the tile map.
     */
    const map_type &tiles() const { return _tiles; }

    /**
     * Get a reference to the tile map.
     */
    map_type &tiles() { return _tiles; }

    /**
     * Get the name.
     */
    virtual const std::string &name() const override { return _name; }

    /**
     * Sets the name.
     *
     * @param val The new name.
     */
    virtual void set_name(const std::string &val) override { _name = val; }

    /**
     * Gets the size of the tensor.
     */
    size_t size() const { return _size; }

    /**
     * Gets the number of possible tiles, empty and filled.
     */
    size_t grid_size() const { return _grid_size; }

    /**
     * Gets the number of possible tiles along an axis, empty and filled.
     */
    size_t grid_size(int i) const { return _tile_sizes[i].size(); }

    /**
     * Gets the number of filled tiles.
     */
    size_t num_filled() const { return _tiles.size(); }

    virtual bool full_view_of_underlying() const override { return true; }

    /**
     * @brief Get the dimensions.
     */
    virtual size_t dim(int d) const override { return _dims.at(d); }

    virtual Dim<Rank> dims() const override { return _dims; }

    /**
     * Check to see if the given tile has zero size.
     *
     * @param index The index of the tile, as a list of integers.
     * @return True if the tile has at least one dimension of zero, leading to a size of zero, or false if there are no zero dimensions.
     */
    template <std::integral... Index>
        requires(sizeof...(Index) == Rank)
    bool has_zero_size(Index... index) const {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < Rank; i++) {
            if (_tile_sizes[i].at(arr_index[i]) == 0) {
                return true;
            }
        }

        return false;
    }

    /**
     * Check to see if the given tile has zero size.
     *
     * @param index The index of the tile, as a container of integers.
     * @return True if the tile has at least one dimension of zero, leading to a size of zero, or false if there are no zero dimensions.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    bool has_zero_size(const Storage &index) const {
        for (int i = 0; i < Rank; i++) {
            if (_tile_sizes[i].at(index[i]) == 0) {
                return true;
            }
        }

        return false;
    }

    /**
     * Convert.
     */
    operator TensorType() const {
        TensorType out(_dims);
        out.set_name(name());
        out.zero();

        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            std::apply(out, target_combination) = std::apply(*this, target_combination);
        }

        return out;
    }
};

} // namespace tensor_props

template <typename T, size_t Rank>
struct TiledTensor final : public virtual tensor_props::TiledTensorBase<T, Rank, Tensor<T, Rank>>, virtual tensor_props::CoreTensorBase {
  protected:
    void add_tile(std::array<int, Rank> pos) override {
        std::string tile_name = this->_name + " - (";
        Dim<Rank>   dims{};

        for (int i = 0; i < Rank; i++) {
            tile_name += std::to_string(pos[i]);
            dims[i] = this->_tile_sizes[i][pos[i]];
            if (i != Rank - 1) {
                tile_name += ", ";
            }
        }
        tile_name += ")";

        this->_tiles.emplace(pos, dims);
        this->_tiles[pos].set_name(tile_name);
        this->_tiles[pos].zero();
    }

  public:
    TiledTensor() = default;

    template <typename... Sizes>
    TiledTensor(std::string name, Sizes... sizes) : tensor_props::TiledTensorBase<T, Rank, Tensor<T, Rank>>(name, sizes...) {}

    TiledTensor(std::string name, std::initializer_list<int> init) : tensor_props::TiledTensorBase<T, Rank, Tensor<T, Rank>>(name, init) {}

    TiledTensor(const TiledTensor<T, Rank> &other) : tensor_props::TiledTensorBase<T, Rank, Tensor<T, Rank>>(other) {}

    ~TiledTensor() = default;

    size_t dim(int d) const override { return this->_dims[d]; }
};

template <typename T, size_t Rank>
struct TiledTensorView final : public virtual tensor_props::TiledTensorBase<T, Rank, TensorView<T, Rank>>,
                               virtual tensor_props::TensorViewBase<T, Rank, TiledTensor<T, Rank>>,
                               virtual tensor_props::CoreTensorBase {
  private:
    bool _full_view_of_underlying{false};

    void add_tile(std::array<int, Rank> pos) override { throw std::runtime_error("Can't add a tile to a TiledTensorView!"); }

  public:
    TiledTensorView() = default;

    template <typename... Sizes>
    TiledTensorView(std::string name, Sizes... sizes) : tensor_props::TiledTensorBase<T, Rank, TensorView<T, Rank>>(name, sizes...) {}

    TiledTensorView(const TiledTensorView<T, Rank> &other) = default;

    ~TiledTensorView() = default;

    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    void insert_tile(std::array<int, Rank> pos, TensorView<T, Rank> &view) {
        std::lock_guard(*this);
        this->_tiles[pos] = view;
    }

    size_t dim(int d) const override { return this->_dims[d]; }
};

#ifdef __HIP__
template <typename T, size_t Rank>
struct TiledDeviceTensor final : public virtual tensor_props::TiledTensorBase<T, Rank, DeviceTensor<T, Rank>>,
                                 virtual tensor_props::DeviceTensorBase {
  private:
    detail::HostToDeviceMode _mode{detail::DEV_ONLY};

    void add_tile(std::array<int, Rank> pos) override {
        std::string tile_name = this->_name + " - (";
        Dim<Rank>   dims{};

        for (int i = 0; i < Rank; i++) {
            tile_name += std::to_string(pos[i]);
            dims[i] = this->_tile_sizes[i].at(pos[i]);
            if (i != Rank - 1) {
                tile_name += ", ";
            }
        }
        tile_name += ")";

        this->_tiles.emplace(pos, DeviceTensor<T, Rank>(dims, _mode));
        this->_tiles.at(pos).set_name(tile_name);
        this->_tiles.at(pos).zero();
    }

  public:
    TiledDeviceTensor() = default;

    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensor(std::string name, detail::HostToDeviceMode mode, Sizes... sizes)
        : _mode{mode}, tensor_props::TiledTensorBase<T, Rank, DeviceTensor<T, Rank>>(name, sizes...) {}

    TiledDeviceTensor(const TiledDeviceTensor<T, Rank> &other) = default;

    ~TiledDeviceTensor() = default;

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    T operator()(MultiIndex... index) const {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets[i][coords[i]];
        }

        if (this->has_tile(coords)) {
            return std::apply(this->tile(coords), array_ind);
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
    HostDevReference<T> operator()(MultiIndex... index) {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets[i][coords[i]];
        }
        auto &out = this->tile(coords);

        return std::apply(out, array_ind);
    }

    operator Tensor<T, Rank>() const { return (Tensor<T, Rank>)(DeviceTensor<T, Rank>)*this; }

    size_t dim(int d) const override { return this->_dims[d]; }
};

template <typename T, size_t Rank>
struct TiledDeviceTensorView final : public virtual tensor_props::TiledTensorBase<T, Rank, DeviceTensorView<T, Rank>>,
                                     virtual tensor_props::DeviceTensorBase,
                                     virtual tensor_props::TensorViewBase<T, Rank, TiledDeviceTensor<T, Rank>> {
  private:
    bool                     _full_view_of_underlying{false};
    detail::HostToDeviceMode _mode{detail::DEV_ONLY};

    void add_tile(std::array<int, Rank> pos) override { throw std::runtime_error("Can't add a tile to a TiledDeviceTensorView!"); }

  public:
    TiledDeviceTensorView() = default;

    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensorView(std::string name, detail::HostToDeviceMode mode, Sizes... sizes)
        : _mode{mode}, tensor_props::TiledTensorBase<T, Rank, DeviceTensorView<T, Rank>>(name, sizes...) {}

    TiledDeviceTensorView(const TiledDeviceTensorView<T, Rank> &other) = default;

    ~TiledDeviceTensorView() = default;

    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    void insert_tile(std::array<int, Rank> pos, DeviceTensorView<T, Rank> &view) { this->_tiles[pos] = view; }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    T operator()(MultiIndex... index) const {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets[i][coords[i]];
        }

        if (this->has_tile(coords)) {
            return std::apply(this->tile(coords), array_ind);
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
    HostDevReference<T> operator()(MultiIndex... index) {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets[i][coords[i]];
        }
        auto &out = this->tile(coords);

        return std::apply(out, array_ind);
    }

    size_t dim(int d) const override { return this->_dims[d]; }
};

#endif

} // namespace einsums

template <template <typename, size_t> typename TensorType, size_t Rank, typename T>
    requires einsums::RankTiledTensor<TensorType<T, Rank>, Rank, T>
void println(const TensorType<T, Rank> &A, TensorPrintOptions options = {}) {
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

template <template <typename, size_t> typename TensorType, size_t Rank, typename T>
    requires einsums::RankTiledTensor<TensorType<T, Rank>, Rank, T>
void fprintln(FILE *fp, const TensorType<T, Rank> &A, TensorPrintOptions options = {}) {
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

template <template <typename, size_t> typename TensorType, size_t Rank, typename T>
    requires einsums::RankTiledTensor<TensorType<T, Rank>, Rank, T>
void fprintln(std::ostream &os, const TensorType<T, Rank> &A, TensorPrintOptions options = {}) {
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
                fprintln(os, A.tile(tile_index));
            }
        }
    }
}