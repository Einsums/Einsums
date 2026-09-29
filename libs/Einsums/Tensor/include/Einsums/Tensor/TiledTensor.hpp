//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/HashFunctions.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <string>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/Tensor/DeviceTensor.hpp>
#endif

#include <array>
#include <cmath>
#include <concepts>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace einsums {

namespace tensor_base {

/**
 * @struct TiledTensor
 *
 * Represents a tiled tensor. Is a lockable type.
 *
 * @tparam TensorType The underlying storage type.
 * @tparam T The type of data being stored.
 * @tparam Rank The tensor rank.
 */
template <typename T, size_t rank, typename TensorType>
struct TiledTensor : public TiledTensorNoExtra, design_pats::Lockable<std::recursive_mutex>, AlgebraOptimizedTensor {
  public:
    /**
     * @typedef map_type
     *
     * @brief The data type used to hold the subtensors.
     */
    using map_type = typename std::unordered_map<std::array<int, rank>, TensorType, einsums::hashes::container_hash<std::array<int, rank>>>;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef ValueType
     *
     * @brief Represents the type of data stored in this tensor.
     */
    using ValueType = T;

    /**
     * @typedef StoredType
     *
     * @brief The types of tensor this collected tensor stores.
     */
    using StoredType = TensorType;

    /**
     * Create a new empty tiled tensor.
     */
    TiledTensor() : _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{}, _grid_size{0} {}

    /**
     * Create a new empty tiled tensor with the given grid. If only one grid is given, the grid is applied to all dimensions.
     * Otherwise, the number of grids must match the rank.
     *
     * @param name The name of the tensor.
     * @param sizes The grids to apply.
     */
    template <ContainerOrInitializer... Sizes>
        requires(!ContainerOrInitializer<typename Sizes::value_type> && ... && true)
    TiledTensor(std::string name, Sizes const &...sizes);

    /**
     * @brief Create a new empty tiled tensor with the given grid.
     *
     * @param name The name of the tensor.
     * @param sizes The grids to apply.
     */
    template <Container ContainerType>
        requires(Container<typename ContainerType::value_type> && std::is_integral_v<typename ContainerType::value_type::value_type>)
    TiledTensor(std::string name, ContainerType const &sizes);

    /**
     * Copy a tiled tensor.
     *
     * @param other The tensor to be copied.
     */
    TiledTensor(TiledTensor<T, rank, TensorType> const &other);

    /**
     * Copy a tiled tensor from one tensor type to another.
     *
     * @param other The tensor to be copied and converted.
     */
    template <TRTensorConcept<rank, T> OtherTensor>
    TiledTensor(TiledTensor<T, rank, OtherTensor> const &other);

    TiledTensor(TiledTensor &&)
        requires(std::is_move_constructible_v<TensorType>)
    = default;

    virtual ~TiledTensor() = default;

    /**
     * Returns the tile with given coordinates. If the tile is not filled, it will be created.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    TensorType &tile(MultiIndex... index);

    /**
     * Returns the tile with given coordinates. If the tile is not filled, this will throw an error.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    TensorType const &tile(MultiIndex... index) const;

    /**
     * Returns the tile with given coordinates. If the tile is not filled, it will be created.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    TensorType &tile(Storage index);

    /**
     * Returns the tile with given coordinates. If the tile is not filled, this will throw an error.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    TensorType const &tile(Storage index) const;

    /**
     * Returns whether a tile exists at a given position, and if it is filled.
     *
     * @param index The position to check for a tile.
     * @return True if there is a tile and it is initialized at this position. False if there is no tile or it is not initialized.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    bool has_tile(MultiIndex... index) const;

    /**
     * Returns the tile coordinates of a given tensor index.
     *
     * @param index The tensor index to check.
     * @return The coordinates of the tile.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    std::array<int, rank> tile_of(MultiIndex... index) const;

    /**
     * Returns whether a tile exists at a given position, and if it is filled.
     *
     * @param index The position to check for a tile.
     * @return True if there is a tile and it is initialized at this position. False if there is no tile or it is not initialized.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    bool has_tile(Storage index) const;

    /**
     * Returns the tile coordinates of a given tensor index.
     *
     * @param index The tensor index to check.
     * @return The coordinates of the tile.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    std::array<int, rank> tile_of(Storage index) const;

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    T operator()(MultiIndex... index) const;

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    T at(MultiIndex... index) const;

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    T &operator()(MultiIndex... index);

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <typename ContainerType>
        requires requires {
            requires !std::is_integral_v<ContainerType>;
            requires !std::is_same_v<ContainerType, Dim<rank>>;
            requires !std::is_same_v<ContainerType, Stride<rank>>;
            requires !std::is_same_v<ContainerType, Offset<rank>>;
            requires !std::is_same_v<ContainerType, Range>;
        }
    T operator()(ContainerType const &index) const;

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <typename ContainerType>
        requires requires {
            requires !std::is_integral_v<ContainerType>;
            requires !std::is_same_v<ContainerType, Dim<rank>>;
            requires !std::is_same_v<ContainerType, Stride<rank>>;
            requires !std::is_same_v<ContainerType, Offset<rank>>;
            requires !std::is_same_v<ContainerType, Range>;
        }
    T at(ContainerType const &index) const;

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <typename ContainerType>
        requires requires {
            requires !std::is_integral_v<ContainerType>;
            requires !std::is_same_v<ContainerType, Dim<rank>>;
            requires !std::is_same_v<ContainerType, Stride<rank>>;
            requires !std::is_same_v<ContainerType, Offset<rank>>;
            requires !std::is_same_v<ContainerType, Range>;
        }
    T &operator()(ContainerType const &index);

    /**
     * Sets all entries in the tensor to zero. This clears all tiles. There will be no more tiles after this.
     */
    void zero();

    /**
     * Sets all entries in the tensor to zero. This keeps all tiles, just calls zero on the tensors.
     */
    void zero_no_clear();

    /**
     * Sets all entries to the given value. Initializes all tiles, unless zero is given.
     * If zero is passed, calls @ref zero
     *
     * @param value The value to broadcast.
     */
    void set_all(T value);

    /**
     * Sets all entries to the given value. If a tile does not exist, it is ignored.
     *
     * @param value The value to broadcast.
     */
    void set_all_existing(T value);

    /**
     * @brief Copy assignment.
     *
     * @param copy The tensor to copy.
     */
    TiledTensor<T, rank, TensorType> &operator=(TiledTensor<T, rank, TensorType> const &copy);

    /**
     * @brief Move assignment.
     *
     * @param move The tensor to move.
     */
    TiledTensor<T, rank, TensorType> &operator=(TiledTensor<T, rank, TensorType> &&move);

    /**
     * @brief Copy assignment from a different kind of tiled tensor.
     *
     * @param copy The tensor to copy.
     */
    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<TiledTensor<T, rank, TensorType>, TensorOther>)
    TiledTensor<T, rank, TensorType> &operator=(TensorOther const &copy);

    /**
     * @brief Set all occupied tensors with the given value.
     *
     * @param value The value to fill the tensors with.
     */
    TiledTensor &operator=(T value);

    /**
     * @brief Add a scalar to every tensor.
     *
     * @param value The value to add.
     */
    TiledTensor &operator+=(T value);

    /**
     * @brief Subtract a scalar from every tensor.
     *
     * @param value The value to subtract.
     */
    TiledTensor &operator-=(T value);

    /**
     * @brief Multiply every tensor by a scalar.
     *
     * @param value The value to multiply.
     */
    TiledTensor &operator*=(T value);

    /**
     * @brief Divide every tensor by a scalar.
     *
     * @param value The value to divide by.
     */
    TiledTensor &operator/=(T value);

    /**
     * @brief Perform in-place addition between two tensors.
     *
     * @param other The tensor to add to this one.
     */
    TiledTensor &operator+=(TiledTensor const &other);

    /**
     * @brief Perform in-place subtraction between two tensors.
     *
     * @param other The tensor to subtract from this one.
     */
    TiledTensor &operator-=(TiledTensor const &other);

    /**
     * @brief Perform in-place multiplication between two tensors.
     *
     * @param other The tensor to multiply this one by.
     */
    TiledTensor &operator*=(TiledTensor const &other);

    /**
     * @brief Perform in-place division between two tensors.
     *
     * If a block is zero in the divisor, then the corresponding block will also be zeroed in the output,
     * rather than setting it to NaN or infinity.
     *
     * @param other The tensor to division this one by.
     */
    TiledTensor &operator/=(TiledTensor const &other);

    /**
     * Returns the tile offsets.
     */
    std::array<std::vector<int>, rank> tile_offsets() const;

    /**
     * Returns the tile offsets along a given dimension.
     *
     * @param i The axis to retrieve.
     *
     */
    std::vector<int> tile_offset(int i = 0) const;

    /**
     * Returns the tile sizes.
     */
    std::array<std::vector<int>, rank> tile_sizes() const;

    /**
     * Returns the tile sizes along a given dimension.
     *
     * @param i The axis to retrieve.
     *
     */
    std::vector<int> tile_size(int i = 0) const;

    /**
     * Get a reference to the tile map.
     */
    map_type const &tiles() const;

    /**
     * Get a reference to the tile map.
     */
    map_type &tiles();

    /**
     * Get the name.
     */
    virtual std::string const &name() const;

    /**
     * Sets the name.
     *
     * @param val The new name.
     */
    virtual void set_name(std::string const &val);

    /**
     * Gets the size of the tensor.
     */
    size_t size() const;

    /**
     * Gets the number of possible tiles, empty and filled.
     */
    size_t grid_size() const;

    /**
     * Gets the number of possible tiles along an axis, empty and filled.
     */
    size_t grid_size(int i) const;

    /**
     * Gets the number of filled tiles.
     */
    size_t num_filled() const;

    /**
     * @brief Indicates whether the tensor sees all of the underlying elements, or could if all blocks were filled.
     */
    virtual bool full_view_of_underlying() const;

    /**
     * @brief Get the dimension along a given axis.
     *
     * @param d The axis to query.
     */
    size_t dim(int d) const;

    /**
     * @brief Get the dimensions
     */
    Dim<rank> dims() const;

    /**
     * Check to see if the given tile has zero size.
     *
     * @param index The index of the tile, as a list of integers.
     * @return True if the tile has at least one dimension of zero, leading to a size of zero, or false if there are no zero dimensions.
     */
    template <std::integral... Index>
        requires(sizeof...(Index) == rank)
    bool has_zero_size(Index... index) const;

    /**
     * Check to see if the given tile has zero size.
     *
     * @param index The index of the tile, as a container of integers.
     * @return True if the tile has at least one dimension of zero, leading to a size of zero, or false if there are no zero dimensions.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    bool has_zero_size(Storage const &index) const;

    /**
     * Convert to the underlying tensor type.
     */
    operator TensorType() const;

  protected:
    /**
     * @property _tile_offsets
     *
     * @brief A list containing the positions along the axes that each tile starts.
     */
    /**
     * @property _tile_sizes
     *
     * @brief A list of the lengths of the tiles along the axes.
     */
    std::array<std::vector<int>, rank> _tile_offsets, _tile_sizes;

    /**
     * @property _tiles
     *
     * @brief The map containing the tiles.
     */
    map_type _tiles;

    /**
     * @property _dims
     *
     * @brief The overall dimensions of the tensor.
     */
    Dim<rank> _dims;

    /**
     * @property _size
     *
     * @brief The total number of elements in the tensor, including the ignored zeros.
     */
    /**
     * @property _grid_size
     *
     * @brief The number of possible tile positions within the grid.
     */
    size_t _size, _grid_size;

    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed)"};

    /**
     * Add a tile using the underlying type's preferred method. Also gives it a name.
     *
     * @param pos The position in the grid to place the tile.
     */
    virtual void add_tile(std::array<int, rank> const &pos) = 0;

    template <typename TOther, size_t RankOther, typename TensorTypeOther>
    friend struct TiledTensorBase;
};

} // namespace tensor_base

/**
 * @struct TiledTensor
 *
 * @brief Holds a tile-wise sparse tensor.
 *
 * Tensors of this class have large blocks that are rigorously zero. These blocks need to line up on a grid.
 */
template <typename T, size_t Rank>
struct TiledTensor final : public tensor_base::TiledTensor<T, Rank, einsums::Tensor<T, Rank>>, tensor_base::CoreTensor {
  protected:
    /**
     * @brief Construct a new tile in the set of tiles at the given position.
     *
     * @param pos The position of the tile to create.
     */
    void add_tile(std::array<int, Rank> const &pos) override {
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

        this->_tiles.emplace(pos, dims);
        this->_tiles.at(pos).set_name(tile_name);
        this->_tiles.at(pos).zero();
    }

  public:
    TiledTensor() = default;

    /**
     * @brief Create a new tiled tensor with the given name and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
    TiledTensor(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, Tensor<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * @param other The tensor to copy.
     */
    TiledTensor(TiledTensor<T, Rank> const &other) : tensor_base::TiledTensor<T, Rank, Tensor<T, Rank>>(other) {}

    /**
     * @brief Copy cast constructor.
     *
     * This constructor copies data from a tiled tensor that is not just a TiledTensor. This includes
     * TiledTensorView, TiledDeviceTensor, and others.
     *
     * @param other The tensor to copy.
     */
    template <TiledTensorConcept OtherTensor>
        requires(SameUnderlyingAndRank<TiledTensor<T, Rank>, OtherTensor>)
    TiledTensor(OtherTensor const &other) : tensor_base::TiledTensor<T, Rank, Tensor<T, Rank>>(other) {}

    TiledTensor(TiledTensor &&) = default;

    ~TiledTensor() = default;

    /**
     * @brief Copy assignment.
     *
     * This copies the data from the other tensor into this one. The other tensor should be
     * some sort of tiled tensor as well.
     *
     * @param copy The tensor to copy.
     */
    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<TiledTensor<T, Rank>, TensorOther>)
    TiledTensor<T, Rank> &operator=(TensorOther const &copy) {
        this->zero();
        this->_tile_sizes   = copy.tile_sizes();
        this->_tile_offsets = copy.tile_offsets();
        this->_dims         = copy.dims();
        this->_name         = copy.name();
        this->_size         = copy.size();
        this->_grid_size    = copy.grid_size();

        for (auto const &tile : copy.tiles()) {
            add_tile(tile.first);
            this->_tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    /**
     * @brief Copy assignment.
     *
     * This copies the data from the other tensor into this one. The other tensor should be
     * some sort of tiled tensor as well.
     *
     * @param copy The tensor to copy.
     */
    TiledTensor<T, Rank> &operator=(TiledTensor<T, Rank> &&copy) = default;
};

/**
 * @struct TiledTensorView
 *
 * @brief Tensors of this class hold views of the tiles of a tiled tensor.
 *
 * Since views of block tensors are not guaranteed to be truly block diagonal, TiledTensorViews also hold
 * views of BlockTensors when the view is not hypersquare.
 */
template <typename T, size_t Rank>
struct TiledTensorView final : public tensor_base::TiledTensor<T, Rank, einsums::TensorView<T, Rank>>, tensor_base::CoreTensor {
  private:
    /**
     * @property _full_view_of_underlying
     *
     * @brief Indicates whether this view can see all of the elements in the base tensor.
     */
    bool _full_view_of_underlying{false};

    /**
     * @brief Tries to add a tile to the tensor, but it can't.
     *
     * As of the current version, modification of the underlying structure of a TiledTensorView is
     * not allowed. This is because currently, TiledTensorViews are kind of scuffed in that
     * their structure is desynchronized from the tensor they view. This may change in the future.
     */
    void add_tile(std::array<int, Rank> const &) override {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "Can't add a tile to a TiledTensorView!");
    }

  public:
    /**
     * @typedef underlying_type
     *
     * @brief Represents the kind of tensor this object views.
     */
    using underlying_type = TiledTensor<T, Rank>;

    TiledTensorView() = default;

    /**
     * @brief Create an empty view with the given name and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
    TiledTensorView(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, einsums::TensorView<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * On exit, the structure of the copied tensor is copied. This means that the new tensor will view the same tensor
     * as the tensor being copied. Modifications to one will modify the other.
     *
     * @param other The tensor view to copy.
     */
    TiledTensorView(TiledTensorView<T, Rank> const &other) = default;

    ~TiledTensorView() = default;

    /**
     * @brief Checks to see if the view sees all of the data in the tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    /**
     * @brief Add a tile to the view.
     *
     * This does not add a tile to the viewed tensor, only to the view.
     */
    void insert_tile(std::array<int, Rank> pos, einsums::TensorView<T, Rank> &&view) {
        std::lock_guard lock(*this);
        this->_tiles.emplace(pos, view);
    }
};

#ifdef EINSUMS_COMPUTE_CODE
template <typename T, size_t Rank>
struct TiledDeviceTensor final : public tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>, tensor_base::DeviceTensorBase {
  private:
    /**
     * @property _mode
     *
     * @brief The storage mode to use for the subtensors.
     */
    detail::HostToDeviceMode _mode{detail::DEV_ONLY};

    /**
     * @brief Create a new uninitialized tensor at the given position.
     *
     * @param pos The position to put the new tensor.
     */
    void add_tile(std::array<int, Rank> const &pos) override {
        auto        lock      = std::lock_guard(*this);
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

        this->_tiles.emplace(pos, einsums::DeviceTensor<T, Rank>(dims, _mode));
        this->_tiles.at(pos).set_name(tile_name);
        this->_tiles.at(pos).zero();
    }

  public:
    TiledDeviceTensor() = default;

    /**
     * @brief Create a new tiled tensor with the given name, storage mode, and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param mode The storage mode for the tensors.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensor(std::string name, detail::HostToDeviceMode mode, Sizes &&...sizes)
        : _mode{mode}, tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Create a new tiled tensor with the given name and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensor(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * @param other The tensor to copy.
     */
    TiledDeviceTensor(TiledDeviceTensor<T, Rank> const &other) = default;

    /**
     * @brief Copy the data from another tiled tensor into this one.
     *
     * The parameter should be some type of tiled tensor or tiled tensor view.
     *
     * @param other The tensor to copy.
     */
    template <RankTiledTensor<Rank, T> OtherType>
    TiledDeviceTensor(OtherType const &other) : tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(other) {}

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
            array_ind[i] -= this->_tile_offsets[i].at(coords[i]);
        }

        if (this->has_tile(coords)) {
            return subscript_tensor(this->tile(coords), array_ind);
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
            array_ind[i] -= this->_tile_offsets[i].at(coords[i]);
        }
        auto &out = this->tile(coords);

        return subscript_tensor(out, array_ind);
    }

    /**
     * @brief Indexes into the tensor.
     *
     * If the appropriate tile does not exist, this will return zero.
     *
     * @param index The index for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) const -> T {
        return std::apply(*this, index);
    }

    /**
     * @brief Indexes into the tensor.
     *
     * If the appropriate tile does not exist, it will be created, zeroed, and the value returned.
     *
     * @param index The index to use for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) -> HostDevReference<T> {
        return std::apply(*this, index);
    }

    /**
     * @brief Copy assignment.
     *
     * Copies the data from another tiled tensor into this one.
     *
     * @param copy The tensor to copy.
     */
    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<TiledDeviceTensor<T, Rank>, TensorOther>)
    TiledDeviceTensor<T, Rank> &operator=(TensorOther const &copy) {
        this->zero();
        this->_tile_sizes   = copy.tile_sizes();
        this->_tile_offsets = copy.tile_offsets();
        this->_dims         = copy.dims();
        this->_name         = copy.name();
        this->_size         = copy.size();
        this->_grid_size    = copy.grid_size();

        for (auto const &tile : copy.tiles()) {
            add_tile(tile.first);
            this->_tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    /**
     * @brief Cast to normal tensor.
     */
    operator einsums::Tensor<T, Rank>() const { return (einsums::Tensor<T, Rank>)(einsums::DeviceTensor<T, Rank>)*this; }
};

template <typename T, size_t Rank>
struct TiledDeviceTensorView final : public tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>, tensor_base::DeviceTensorBase {
  public:
    /**
     * @typedef underlying_type
     *
     * @brief The type of tensor this object views.
     */
    using underlying_type = TiledDeviceTensor<T, Rank>;

    TiledDeviceTensorView() = default;

    /**
     * @brief Construct a new tensor view with the given name, storage mode, and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param mode The storage mode for the tensors.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensorView(std::string name, detail::HostToDeviceMode mode, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Construct a new tensor view with the given name, storage mode, and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param mode The storage mode for the tensors.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensorView(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * Only the internal structure is truly copied. The views contained in this tensor view will still point to the
     * same place as the copied tensor, meaning updates to one will update the other.
     *
     * @param other The tensor view to copy.
     */
    TiledDeviceTensorView(TiledDeviceTensorView<T, Rank> const &other) = default;

    /**
     * @brief Create a copy of the core tiled tensor on the GPU.
     *
     * @param other The tensor to copy.
     */
    TiledDeviceTensorView(TiledTensor<T, Rank> &other)
        : tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>(other.name(), other.tile_sizes()) {
        for (auto &tile : other.tiles()) {
            this->_tiles.emplace(tile.first, tile.second);
        }
    }

    ~TiledDeviceTensorView() = default;

    /**
     * @brief Indicates whether the view can see all of the data of the original tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    /**
     * @brief Add a tile to the structure of this tensor.
     *
     * @param pos The position to put the view.
     * @param view The view to add to the tensor.
     */
    void insert_tile(std::array<int, Rank> pos, DeviceTensorView<T, Rank> &&view) {
        std::lock_guard lock(*this);
        this->_tiles.emplace(pos, view);
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
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets.at(i).at(coords[i]);
        }

        if (this->has_tile(coords)) {
            return subscript_tensor(this->tile(coords), array_ind);
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
            array_ind[i] -= this->_tile_offsets.at(i).at(coords[i]);
        }
        auto &out = this->tile(coords);

        return subscript_tensor(out, array_ind);
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) const -> T {
        return std::apply(*this, index);
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) -> HostDevReference<T> {
        return std::apply(*this, index);
    }

  private:
    /**
     * @property _full_view_of_underlying
     *
     * @brief Indicates whether the view can see all of the elements of the underlying tensor.
     */
    bool _full_view_of_underlying{false};

    /**
     * @brief Tries to add a tile to the tensor, but it can't.
     *
     * As of the current version, modification of the underlying structure of a TiledTensorView is
     * not allowed. This is because currently, TiledTensorViews are kind of scuffed in that
     * their structure is desynchronized from the tensor they view. This may change in the future.
     */
    void add_tile(std::array<int, Rank> const &pos) override {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "Can't add a tile to a TiledDeviceTensorView!");
    }
};

TENSOR_EXPORT(TiledDeviceTensor)
TENSOR_EXPORT(TiledDeviceTensorView)

#endif

TENSOR_EXPORT(TiledTensor)
TENSOR_EXPORT(TiledTensorView)

/**
 * Prints a TiledTensor to standard output.
 */
template <einsums::TiledTensorConcept TensorType>
void println(TensorType const &A, TensorPrintOptions options = {});

/**
 * Prints a TiledTensor to a file pointer.
 */
template <einsums::TiledTensorConcept TensorType>
void fprintln(FILE *fp, TensorType const &A, TensorPrintOptions options = {});

/**
 * Prints a TiledTensor to an output stream.
 */
template <einsums::TiledTensorConcept TensorType>
void fprintln(std::ostream &os, TensorType const &A, TensorPrintOptions options = {});

} // namespace einsums

#include <Einsums/Tensor/Backends/TiledTensorBase.hpp>