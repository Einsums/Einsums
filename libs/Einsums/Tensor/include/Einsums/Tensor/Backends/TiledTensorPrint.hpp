//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Tensor/TiledTensor.hpp>

namespace einsums {

template <einsums::TiledTensorConcept TensorType>
void println(TensorType const &A, TensorPrintOptions options) {
    using T               = typename TensorType::ValueType;
    constexpr size_t Rank = TensorType::Rank;
    println("Name: {}", A.name());
    {
        print::Indent const indent{};
        println("Tiled Tensor");
        println("Data Type: {}", type_name<T>());

        if constexpr (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            println("Dims{{{}}}", oss.str().c_str());
        }

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
                println(A.tile(tile_index), options);
            }
        }
    }
}

template <einsums::TiledTensorConcept TensorType>
void fprintln(FILE *fp, TensorType const &A, TensorPrintOptions options) {
    using T               = typename TensorType::ValueType;
    constexpr size_t Rank = TensorType::Rank;
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(fp, "Tiled Tensor");
        fprintln(fp, "Data Type: {}", type_name<T>());

        if constexpr (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(fp, "Dims{{{}}}", oss.str().c_str());
        }

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
                fprintln(fp, A.tile(tile_index)), options;
            }
        }
    }
}

template <einsums::TiledTensorConcept TensorType>
void fprintln(std::ostream &os, TensorType const &A, TensorPrintOptions options) {
    using T               = typename TensorType::ValueType;
    constexpr size_t Rank = TensorType::Rank;
    fprintln(os, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(os, "Tiled Tensor");
        fprintln(os, "Data Type: {}", type_name<T>());

        if constexpr (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(os, "Dims{{{}}}", oss.str().c_str());
        }

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
                fprintln(os, A.tile(tile_index), options);
            }
        }
    }
}

} // namespace einsums
