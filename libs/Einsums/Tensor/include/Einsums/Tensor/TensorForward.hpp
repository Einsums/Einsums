//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator.hpp>

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
template <typename T, size_t Rank>
struct Tensor;

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

template <typename T>
struct RuntimeTensor;

template <typename T>
struct RuntimeTensorView;

template <typename T>
using VectorData = BufferVector<T>;
#endif

} // namespace einsums

#if !defined(EINSUMS_WINDOWS)
/**
 * @def TENSOR_EXPORT_TR
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_EXPORT_TR(tensortype, type, rank) extern template class EINSUMS_EXPORT tensortype<type, rank>;

/**
 * @def TENSOR_EXPORT_RANK
 *
 * Creates exported template declarations for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to declare.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_EXPORT_RANK(tensortype, rank)                                                                                           \
        TENSOR_EXPORT_TR(tensortype, float, rank)                                                                                          \
        TENSOR_EXPORT_TR(tensortype, double, rank)                                                                                         \
        TENSOR_EXPORT_TR(tensortype, std::complex<float>, rank)                                                                            \
        TENSOR_EXPORT_TR(tensortype, std::complex<double>, rank)

/**
 * @def TENSOR_EXPORT
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 */
#    define TENSOR_EXPORT(tensortype)                                                                                                      \
        TENSOR_EXPORT_RANK(tensortype, 1)                                                                                                  \
        TENSOR_EXPORT_RANK(tensortype, 2)                                                                                                  \
        TENSOR_EXPORT_RANK(tensortype, 3)                                                                                                  \
        TENSOR_EXPORT_RANK(tensortype, 4)

/**
 * @def TENSOR_DEFINE_TR
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_TR(tensortype, type, rank) template class tensortype<type, rank>;

/**
 * @def TENSOR_DEFINE_RANK
 *
 * Creates exported template definitions for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to define.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_RANK(tensortype, rank)                                                                                           \
        TENSOR_DEFINE_TR(tensortype, float, rank)                                                                                          \
        TENSOR_DEFINE_TR(tensortype, double, rank)                                                                                         \
        TENSOR_DEFINE_TR(tensortype, std::complex<float>, rank)                                                                            \
        TENSOR_DEFINE_TR(tensortype, std::complex<double>, rank)

/**
 * @def TENSOR_DEFINE
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 */
#    define TENSOR_DEFINE(tensortype)                                                                                                      \
        TENSOR_DEFINE_RANK(tensortype, 1)                                                                                                  \
        TENSOR_DEFINE_RANK(tensortype, 2)                                                                                                  \
        TENSOR_DEFINE_RANK(tensortype, 3)                                                                                                  \
        TENSOR_DEFINE_RANK(tensortype, 4)

/**
 * @def TENSOR_EXPORT_TR_DISK_VIEW
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_EXPORT_TR_DISK_VIEW(tensortype, type, view_rank, rank)                                                                  \
        extern template class EINSUMS_EXPORT tensortype<type, view_rank, rank>;

/**
 * @def TENSOR_EXPORT_RANK_DISK_VIEW
 *
 * Creates exported template declarations for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to declare.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, view_rank, rank)                                                                      \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, float, view_rank, rank)                                                                     \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, double, view_rank, rank)                                                                    \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, std::complex<float>, view_rank, rank)                                                       \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, std::complex<double>, view_rank, rank)

/**
 * @def TENSOR_EXPORT_RANK2_DISK_VIEW
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, rank)                                                                                \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 1, rank)                                                                                  \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 2, rank)                                                                                  \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 3, rank)                                                                                  \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 4, rank)

/**
 * @def TENSOR_EXPORT_DISK_VIEW
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks and view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 */
#    define TENSOR_EXPORT_DISK_VIEW(tensortype)                                                                                            \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 1)                                                                                       \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 2)                                                                                       \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 3)                                                                                       \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 4)

/**
 * @def TENSOR_DEFINE_TR_DISK_VIEW
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 * @param view_rank The rank of the view
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_TR_DISK_VIEW(tensortype, type, view_rank, rank) template class tensortype<type, view_rank, rank>;

/**
 * @def TENSOR_DEFINE_RANK_DISK_VIEW
 *
 * Creates exported template definitions for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to define.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, view_rank, rank)                                                                      \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, float, view_rank, rank)                                                                     \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, double, view_rank, rank)                                                                    \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, std::complex<float>, view_rank, rank)                                                       \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, std::complex<double>, view_rank, rank)

/**
 * @def TENSOR_DEFINE_RANK2_DISK_VIEW
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, rank)                                                                                \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 1, rank)                                                                                  \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 2, rank)                                                                                  \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 3, rank)                                                                                  \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 4, rank)

/**
 * @def TENSOR_DEFINE_DISK_VIEW
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks and view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 */
#    define TENSOR_DEFINE_DISK_VIEW(tensortype)                                                                                            \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 1)                                                                                       \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 2)                                                                                       \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 3)                                                                                       \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 4)

#else

#    define TENSOR_EXPORT_TR(tensortype, type, rank)
#    define TENSOR_EXPORT_RANK(tensortype, rank)
#    define TENSOR_EXPORT(tensortype)
#    define TENSOR_DEFINE_TR(tensortype, type, rank)
#    define TENSOR_DEFINE_RANK(tensortype, rank)
#    define TENSOR_DEFINE(tensortype)
#    define TENSOR_EXPORT_TR_DISK_VIEW(tensortype, type, view_rank, rank)
#    define TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, view_rank, rank)
#    define TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, rank)
#    define TENSOR_EXPORT_DISK_VIEW(tensortype)
#    define TENSOR_DEFINE_TR_DISK_VIEW(tensortype, type, view_rank, rank)
#    define TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, view_rank, rank)
#    define TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, rank)
#    define TENSOR_DEFINE_DISK_VIEW(tensortype)

#endif