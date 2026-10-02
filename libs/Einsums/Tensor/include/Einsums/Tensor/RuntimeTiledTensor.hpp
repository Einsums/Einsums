//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

#include <cstdlib>
#include <string>
#include <vector>

namespace einsums {

/**
 * @struct TiledTensor
 *
 * @brief Holds a tile-wise sparse tensor.
 *
 * Tensors of this class have large blocks that are rigorously zero. These blocks need to line up on a grid.
 */
template <typename T>
struct RuntimeTiledTensor final : public tensor_base::TiledTensor<T, einsums::RuntimeTensor<T>, std::vector<size_t>>,
                                  tensor_base::CoreTensor {
  protected:
    /**
     * @brief Construct a new tile in the set of tiles at the given position.
     *
     * @param pos The position of the tile to create.
     */
    void add_tile(std::vector<size_t> const &pos) override {
        EINSUMS_ASSERT(pos.size() == this->_rank);
        std::string         tile_name = this->_name + " - (";
        std::vector<size_t> dims(this->_rank);

        for (int i = 0; i < this->_rank; i++) {
            tile_name += std::to_string(pos[i]);
            dims[i] = this->_tile_sizes[i].at(pos[i]);
            if (i != this->_rank - 1) {
                tile_name += ", ";
            }
        }
        tile_name += ")";

        this->_tiles.emplace(pos, dims);
        this->_tiles.at(pos).set_name(tile_name);
        this->_tiles.at(pos).zero();
    }

  public:
    using typename tensor_base::TiledTensor<T, RuntimeTensor<T>, std::vector<size_t>>::ValueType;
    using typename tensor_base::TiledTensor<T, RuntimeTensor<T>, std::vector<size_t>>::StoredType;

    using tensor_base::TiledTensor<T, RuntimeTensor<T>, std::vector<size_t>>::TiledTensor;

    ~RuntimeTiledTensor() = default;
};

/**
 * @struct TiledTensorView
 *
 * @brief Tensors of this class hold views of the tiles of a tiled tensor.
 *
 * Since views of block tensors are not guaranteed to be truly block diagonal, TiledTensorViews also hold
 * views of BlockTensors when the view is not hypersquare.
 */
template <typename T>
struct RuntimeTiledTensorView final : public tensor_base::TiledTensor<T, RuntimeTensorView<T>, std::vector<size_t>>,
                                      tensor_base::CoreTensor {
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
    void add_tile(std::vector<size_t> const &) override {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "Can't add a tile to a TiledTensorView!");
    }

  public:
    /**
     * @typedef underlying_type
     *
     * @brief Represents the kind of tensor this object views.
     */
    using underlying_type = RuntimeTiledTensor<T>;

    using typename tensor_base::TiledTensor<T, RuntimeTensorView<T>, std::vector<size_t>>::ValueType;
    using typename tensor_base::TiledTensor<T, RuntimeTensorView<T>, std::vector<size_t>>::StoredType;

    using tensor_base::TiledTensor<T, RuntimeTensorView<T>, std::vector<size_t>>::TiledTensor;

    RuntimeTiledTensorView() = default;

    ~RuntimeTiledTensorView() = default;

    /**
     * @brief Checks to see if the view sees all of the data in the tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    /**
     * @brief Add a tile to the view.
     *
     * This does not add a tile to the viewed tensor, only to the view.
     */
    void insert_tile(std::vector<size_t> const &pos, RuntimeTensorView<T> &&view) {
        std::lock_guard lock(*this);
        this->_tiles.emplace(pos, view);
    }

    /**
     * @brief Add a tile to the view.
     *
     * This does not add a tile to the viewed tensor, only to the view.
     */
    void insert_tile(std::vector<size_t> const &pos, RuntimeTensorView<T> const &view) {
        std::lock_guard lock(*this);
        this->_tiles.insert(std::pair(pos, view));
    }
};

} // namespace einsums
