//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/Config.hpp>

#include <complex>
#include <memory>
#include <mutex>

#if defined(EINSUMS_COMPUTE_CODE)
#    include <hip/hip_complex.h>
#endif

namespace einsums::tensor_base {


#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @struct DeviceTypedTensor
 *
 * Represents a tensor that stores different data types on the host and the device.
 * By default, if the host stores complex<float> or complex<double>, then it converts
 * it to hipFloatComplex or hipDoubleComplex. The two types should have the same storage size.
 * @tparam HostT The host data type.
 * @tparam DevT The device type.
 */
template <typename HostT, typename DevT = void>
    requires(std::is_void_v<DevT> || sizeof(HostT) == sizeof(DevT))
struct DeviceTypedTensor {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype =
        std::conditional_t<std::is_void_v<DevT>,
                           std::conditional_t<std::is_same_v<HostT, std::complex<float>>, hipFloatComplex,
                                              std::conditional_t<std::is_same_v<HostT, std::complex<double>>, hipDoubleComplex, HostT>>,
                           DevT>;

    /**
     * @typedef host_datatype
     *
     * @brief The datatype that the host sees.
     */
    using host_datatype = HostT;
};
#endif

/*==================
 * Location-based.
 *==================*/

/**
 * @struct CoreTensor
 *
 * @brief Represents a tensor only available to the core.
 */
struct CoreTensor {

};

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @struct DeviceTensor
 *
 * @brief Represents a tensor available to graphics hardware.
 */
struct DeviceTensorBase {

};
#endif

/**
 * @struct DiskTensor
 *
 * @brief Represents a tensor stored on disk.
 */
struct DiskTensor {

};

/*===================
 * Other properties.
 *===================*/

/**
 * @struct TiledTensorNoExtra
 *
 * @brief Specifies that a tensor is a tiled tensor without needing to specify type parameters.
 *
 * Only used internally. Use TiledTensorBase in your code.
 */
struct TiledTensorNoExtra {

};

#ifndef DOXYGEN
// Large class. See TiledTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct TiledTensor;
#endif

/**
 * @struct BlockTensorNoExtra
 *
 * @brief Specifies that a tensor is a block tensor. Internal use only. Use BlockTensorBase instead.
 *
 * Specifies that a tensor is a block tensor without needing template parameters. Internal use only.
 * Use BlockTensorBase in your code.
 */
struct BlockTensorNoExtra {

};

// Large class. See BlockTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct BlockTensor;

/**
 * @struct AlgebraOptimizedTensor
 *
 * @brief Specifies that the tensor type can be used by einsum to select different routines other than the generic algorithm.
 */
struct AlgebraOptimizedTensor {

};

class RuntimeTensorNoType {};

class RuntimeTensorViewNoType {};

} // namespace einsums::tensor_base