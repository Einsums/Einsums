//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

#include "Einsums/Errors/Error.hpp"

namespace einsums {

template <CoreBlockTensorConcept TensorType>
auto apply_view(TensorType const &tensor, std::vector<Range> const &spec) {
    if (spec.size() > tensor.num_blocks()) {
        EINSUMS_THROW_EXCEPTION(num_argument_error, "Can not apply view specification to block tensor. Incorrect number of indices given.");
    }

    std::vector<int64_t> sizes(spec.size());

    for (int i = 0; i < spec.size(); i++) {
        if (spec[i][1] < 0 && tensor.block_dim(i) != 0) {
            sizes[i] = tensor.block_dim(i) + spec[i][1] - spec[i][0];
        } else if (tensor.block_dim(i) == 0) {
            sizes[i] = 0;
        } else {
            sizes[i] = spec[i][1] - spec[i][0];
        }
    }

    auto out = TiledTensorView<typename TensorType::ValueType, TensorType::Rank>("unnamed view", sizes);

    std::array<int, TensorType::Rank> position;

    auto index = std::array<Range, TensorType::Rank>();

    for (int i = 0; i < spec.size(); i++) {
        position.fill((int)i);

        index.fill(spec.at(i));

        out.insert_tile(position, std::apply(tensor[i], index));
    }

    return out;
}

template <CoreBlockTensorConcept TensorType, typename... ViewSpec>
    requires requires {
        requires sizeof...(ViewSpec) == TensorType::Rank;
        requires sizeof...(ViewSpec) > 1;
        requires(std::is_same_v<ViewSpec, std::vector<Range>> && ...);
    }
auto apply_view(TensorType const &tensor, ViewSpec &&...spec) {
    if (((spec.size() > tensor.num_blocks()) || ...)) {
        EINSUMS_THROW_EXCEPTION(num_argument_error, "Can not apply view specification to block tensor. Incorrect number of indices given.");
    }

    auto spec_array = std::array{std::forward<ViewSpec>(spec)...};

    std::array<std::vector<int>, TensorType::Rank> sizes;

    for (int i = 0; i < spec_array.size(); i++) {
        sizes[i].resize(spec_array[i].size());
        for (int j = 0; j < spec_array[i].size(); j++) {
            if (spec_array[i][j][1] < 0 && tensor.block_dim(j) != 0) {
                sizes[i][j] = tensor.block_dim(j) + spec_array[i][j][1] - spec_array[i][j][0];
            } else if (tensor.block_dim(j) == 0) {
                sizes[i][j] = 0;
            } else {
                sizes[i][j] = spec_array[i][j][1] - spec_array[i][j][0];
            }
        }
    }

    auto out = TiledTensorView<typename TensorType::ValueType, TensorType::Rank>("unnamed view", sizes);

    std::array<int, TensorType::Rank> position;

    auto index = std::array<Range, TensorType::Rank>();

    for (int i = 0; i < tensor.num_blocks(); i++) {
        position.fill((int)i);

        for (int j = 0; j < index.size(); j++) {
            index[j] = spec_array[j][i];
        }

        out.insert_tile(position, std::apply(tensor[i], index));
    }

    return out;
}

template <CoreTiledTensorConcept TensorType, typename... ViewSpec>
    requires requires {
        requires sizeof...(ViewSpec) == TensorType::Rank;
        requires sizeof...(ViewSpec) > 1;
        requires(std::is_same_v<ViewSpec, std::vector<Range>> && ...);
    }
auto apply_view(TensorType const &tensor, ViewSpec &&...spec) {
    auto spec_array = std::array{std::forward<ViewSpec>(spec)...};

    for(int i = 0; i < TensorType::Rank; i++) {
        if (spec_array[i].size() > tensor.grid_size(i)) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Can not apply view specification to tiled tensor. Incorrect number of indices given.");
        }
    }
    
    std::array<std::vector<int>, TensorType::Rank> sizes;

    for (int i = 0; i < spec_array.size(); i++) {
        sizes[i].resize(spec_array[i].size());
        for (int j = 0; j < spec_array[i].size(); j++) {
            if (spec_array[i][j][1] < 0 && tensor.tile_size(i)[j] != 0) {
                sizes[i][j] = tensor.tile_size(i)[j] + spec_array[i][j][1] - spec_array[i][j][0];
            } else if (tensor.tile_size(i)[j] == 0) {
                sizes[i][j] = 0;
            } else {
                sizes[i][j] = spec_array[i][j][1] - spec_array[i][j][0];
            }
        }
    }

    auto out = TiledTensorView<typename TensorType::ValueType, TensorType::Rank>("unnamed view", sizes);

    auto index = std::array<Range, TensorType::Rank>();

    for (auto const &pair : tensor.tiles()) {

        for (int j = 0; j < index.size(); j++) {
            index[j] = spec_array[j][pair.first[j]];
        }

        out.insert_tile(pair.first, std::apply(pair.second, index));
    }

    return out;
}

} // namespace einsums