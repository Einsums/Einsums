//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>

#include <cstddef>
#include <vector>

namespace einsums {

/**
 * @struct TensorPrintOptions
 * @brief Represents options and default options for printing tensors.
 */
struct TensorPrintOptions {
    /**
     * @var width
     *
     * How many columns of tensor data are printed per line.
     */
    int width{7};

    /**
     * @var full_output
     *
     * Print the tensor data (true) or just name and data span information (false).
     */
    bool full_output{true};
};

namespace detail {

/**
 * @enum HostToDeviceMode
 *
 * @brief Enum that specifies how device tensors store data and make it available to the GPU.
 */
enum HostToDeviceMode { UNKNOWN, DEV_ONLY, MAPPED, PINNED };

} // namespace detail

#ifndef DOXYGEN
// Forward declarations of tensors.
template <typename T, size_t Rank, typename Alloc>
struct GeneralTensor;

template <typename T, size_t Rank>
using Tensor = GeneralTensor<T, Rank, std::allocator<T>>;

template <typename T, size_t Rank>
using BufferTensor = GeneralTensor<T, Rank, BufferAllocator<T>>;

template <typename T, size_t Rank>
struct BlockTensor;

template <typename T, size_t Rank>
struct TiledTensor;

#    if defined(EINSUMS_COMPUTE_CODE)
template <typename T, size_t Rank>
struct DeviceTensor;

template <typename T, size_t Rank>
struct DeviceTensorView;

template <typename T, size_t Rank>
struct BlockDeviceTensor;

template <typename T, size_t Rank>
struct TiledDeviceTensor;

template <typename T, size_t Rank>
struct TiledDeviceTensorView;
#    endif

template <typename T, size_t Rank>
struct TensorView;

template <typename T, size_t Rank>
struct TiledTensorView;

template <typename T, size_t Rank>
struct DiskView;

template <typename T, size_t Rank>
struct DiskTensor;

template <typename T, typename Alloc>
struct GeneralRuntimeTensor;

template <typename T>
using RuntimeTensor = GeneralRuntimeTensor<T, std::allocator<T>>;

template <typename T>
using BufferRuntimeTensor = GeneralRuntimeTensor<T, BufferAllocator<T>>;

template <typename T>
struct RuntimeTensorView;

template <typename T>
using VectorData = BufferVector<T>;
#endif

} // namespace einsums
