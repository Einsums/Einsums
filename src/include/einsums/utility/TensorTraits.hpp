//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <type_traits>

namespace einsums {

template <typename T, size_t Rank>
struct Tensor;

template <typename T, size_t Rank>
struct TensorView;

template <typename T, size_t Rank>
struct BlockTensor;

template <typename T, size_t Rank>
struct TiledTensor;

template <typename T, size_t Rank>
struct TiledTensorView;

template <typename T, size_t Rank>
struct DiskTensor;

template <typename T, size_t ViewRank, size_t Rank>
struct DiskView;

#ifdef __HIP__
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
#endif

namespace detail {

template <typename D, size_t Rank, typename T>
struct IsIncoreRankTensor
    : public std::bool_constant<std::is_same_v<std::decay_t<D>, Tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, TensorView<T, Rank>> ||
                                std::is_same_v<std::decay_t<D>, BlockTensor<T, Rank>> ||
                                std::is_same_v<std::decay_t<D>, TiledTensor<T, Rank>> ||
                                std::is_same_v<std::decay_t<D>, TiledTensorView<T, Rank>>> {};

template <typename D, size_t Rank, typename T>
inline constexpr bool IsIncoreRankTensorV = IsIncoreRankTensor<D, Rank, T>::value;

#ifndef __HIP__
template <typename D, size_t Rank, typename T>
struct IsBasicTensor
    : public std::bool_constant<std::is_same_v<std::decay_t<D>, Tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, TensorView<T, Rank>>> {
};
#else
template <typename D, size_t Rank, typename T>
struct IsBasicTensor
    : public std::bool_constant<std::is_same_v<std::decay_t<D>, Tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, TensorView<T, Rank>> ||
                                std::is_same_v<std::decay_t<D>, DeviceTensor<T, Rank>> ||
                                std::is_same_v<std::decay_t<D>, DeviceTensorView<T, Rank>>> {};
#endif

template <typename D, size_t Rank, typename T>
constexpr inline bool IsBasicTensorV = IsBasicTensor<D, Rank, T>::value;

template <typename D, size_t Rank, typename T>
struct IsIncoreRankBasicTensor : public std::bool_constant<IsBasicTensorV<D, Rank, T> && IsIncoreRankTensorV<D, Rank, T>> {};

template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankBasicTensorV = IsIncoreRankBasicTensor<D, Rank, T>::value;

// Block tensor tests.
#ifndef __HIP__
template <typename D, size_t Rank, typename T>
struct IsBlockTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, BlockTensor<T, Rank>>> {};
#else
template <typename D, size_t Rank, typename T>
struct IsBlockTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, BlockTensor<T, Rank>> ||
                                                 std::is_same_v<std::decay_t<D>, BlockDeviceTensor<T, Rank>>> {};
#endif

template <typename D, size_t Rank, typename T>
inline constexpr bool IsBlockTensorV = IsBlockTensor<D, Rank, T>::value;

// In-core and block.
template <typename D, size_t Rank, typename T>
struct IsIncoreRankBlockTensor : public std::bool_constant<IsBlockTensorV<D, Rank, T> && IsIncoreRankTensorV<D, Rank, T>> {};

template <typename D, size_t Rank, typename T>
inline constexpr bool IsIncoreRankBlockTensorV = IsIncoreRankBlockTensor<D, Rank, T>::value;

// Tiled tensor tests.
#ifndef __HIP__
template <typename D, size_t Rank, typename T>
struct IsTiledTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, TiledTensor<T, Rank>> ||
                                                 std::is_same_v<std::decay_t<D>, TiledTensorView<T, Rank>>> {};
#else
template <typename D, size_t Rank, typename T>
struct IsTiledTensor
    : public std::bool_constant<
          std::is_same_v<std::decay_t<D>, TiledTensor<T, Rank>> || std::is_same_v<std::decay_t<D>, TiledTensorView<T, Rank>> ||
          std::is_same_v<std::decay_t<D>, TiledDeviceTensor<T, Rank>> || std::is_same_v<std::decay_t<D>, TiledDeviceTensorView<T, Rank>>> {
};
#endif

template <typename D, size_t Rank, typename T>
inline constexpr bool IsTiledTensorV = IsTiledTensor<D, Rank, T>::value;

template <typename D, size_t Rank, typename T>
struct IsIncoreRankTiledTensor : public std::bool_constant<IsTiledTensorV<D, Rank, T> && IsIncoreRankTensorV<D, Rank, T>> {};

template <typename D, size_t Rank, typename T>
inline constexpr bool IsIncoreRankTiledTensorV = IsIncoreRankTiledTensor<D, Rank, T>::value;

template <typename D, size_t Rank, size_t ViewRank = Rank, typename T = double>
struct IsOndiskTensor
    : public std::bool_constant<std::is_same_v<D, DiskTensor<T, Rank>> || std::is_same_v<D, DiskView<T, ViewRank, Rank>>> {};
template <typename D, size_t Rank, size_t ViewRank = Rank, typename T = double>
inline constexpr bool IsOndiskTensorV = IsOndiskTensor<D, Rank, ViewRank, T>::value;

#ifdef __HIP__
/**
 * @struct IsDeviceRankTensor
 *
 * @brief Struct for specifying that a tensor is device compatible.
 */
template <typename D, size_t Rank, typename T>
struct IsDeviceRankTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, DeviceTensor<T, Rank>> ||
                                                      std::is_same_v<std::decay_t<D>, DeviceTensorView<T, Rank>> ||
                                                      std::is_same_v<std::decay_t<D>, BlockDeviceTensor<T, Rank>>> {};

/**
 * @property IsDeviceRankTensorV
 *
 * @brief True if the tensor is device compatible.
 */
template <typename D, size_t Rank, typename T>
inline constexpr bool IsDeviceRankTensorV = IsDeviceRankTensor<D, Rank, T>::value;

/**
 * @struct IsDeviceRankBlockTensor
 *
 * @brief Struct for specifying that a tensor is device compatible, and is block diagonal.
 */
template <typename D, size_t Rank, typename T>
struct IsDeviceRankBlockTensor : public std::bool_constant<IsDeviceRankTensorV<D, Rank, T> && IsBlockTensorV<D, Rank, T>> {};

/**
 * @property IsDeviceRankBlockTensorV
 *
 * @brief True if the tensor is device compatible and is block diagonal.
 */
template <typename D, size_t Rank, typename T>
inline constexpr bool IsDeviceRankBlockTensorV = IsDeviceRankBlockTensor<D, Rank, T>::value;

/**
 * @struct IsDeviceRankTiledTensor
 *
 * @brief Struct for specifying that a tensor is device compatible, and is tiled.
 */
template <typename D, size_t Rank, typename T>
struct IsDeviceRankTiledTensor : public std::bool_constant<IsDeviceRankTensorV<D, Rank, T> && IsTiledTensorV<D, Rank, T>> {};

/**
 * @property IsDeviceRankTiledTensorV
 *
 * @brief True if the tensor is device compatible and is tiled.
 */
template <typename D, size_t Rank, typename T>
inline constexpr bool IsDeviceRankTiledTensorV = IsDeviceRankTiledTensor<D, Rank, T>::value;

template <typename D, size_t Rank, typename T>
struct IsDeviceRankBasicTensor : public std::bool_constant<IsBasicTensorV<D, Rank, T> && IsDeviceRankTensorV<D, Rank, T>> {};

template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankBasicTensorV = IsDeviceRankBasicTensor<D, Rank, T>::value;
#endif

} // namespace detail

/**
 * @brief Concept that requires a tensor to be in core.
 *
 * Example usage:
 *
 * @code
 * template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
 *    requires CoreRankTensor<AType<ADataType, ARank>, 1, ADataType>
 * void sum_square(const AType<ADataType, ARank> &a, RemoveComplexT<ADataType> *scale, RemoveComplexT<ADataType> *sumsq) {}
 * @endcode
 *
 * @tparam Input
 * @tparam Rank
 * @tparam DataType
 */
template <typename Input, size_t Rank, typename DataType = double>
concept CoreRankTensor = detail::IsIncoreRankTensorV<Input, Rank, DataType>;

/**
 * @brief Concept that requires a tensor to be a normal tensor.
 *
 * This allows Tensor, TensorView, DeviceTensor, and DeviceTensorview.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept RankBasicTensor = detail::IsBasicTensorV<Input, Rank, DataType>;

/**
 * @brief Concept that requires a tensor to be a normal tensor and in-core. Allows Tensor and TensorView.
 *
 * This allows Tensor, TensorView, DeviceTensor, and DeviceTensorview.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept CoreRankBasicTensor = detail::IsIncoreRankBasicTensorV<Input, Rank, DataType>;

/**
 * @brief concept that requires a tensor to be a BlockTensor or a BlockDeviceTensor.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept RankBlockTensor = detail::IsBlockTensorV<Input, Rank, DataType>;

/**
 * @brief Concept that requires a tensor to be a TiledTensor, TiledTensorView, TiledDeviceTensor, or TiledDeviceTensorView.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept RankTiledTensor = detail::IsTiledTensorV<Input, Rank, DataType>;

/**
 * @brief Concept that requires a tensor to be in-core and a block tensor. Only allows for BlockTensor.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept CoreRankBlockTensor = detail::IsIncoreRankBlockTensorV<Input, Rank, DataType>;

/**
 * @brief Concept that requires a tensor to be in-core and a tiled tensor. Allows for TiledTensor and TiledTensorView.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept CoreRankTiledTensor = detail::IsIncoreRankTiledTensorV<Input, Rank, DataType>;

/**
 * @brief Concept that requires a tensor to be on-disk.
 */
template <typename Input, size_t Rank, size_t ViewRank = Rank, typename DataType = double>
concept DiskRankTensor = detail::IsOndiskTensorV<Input, Rank, ViewRank, DataType>;

#ifdef __HIP__
/**
 * @concept DeviceRankTensor
 *
 * @brief Concept for testing whether a tensor parameter is available to the GPU.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept DeviceRankTensor = detail::IsDeviceRankTensorV<Input, Rank, DataType>;

/**
 * @concept DeviceRankTensor
 *
 * @brief Concept for testing whether a tensor parameter is available to the GPU.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept DeviceRankBasicTensor = detail::IsDeviceRankBasicTensorV<Input, Rank, DataType>;

/**
 * @concept DeviceRankBlockTensor
 *
 * @brief Concept for testing whether a tensor parameter is available to the GPU and is block diagonal.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept DeviceRankBlockTensor = detail::IsDeviceRankBlockTensorV<Input, Rank, DataType>;

/**
 * @concept DeviceRankBlockTensor
 *
 * @brief Concept for testing whether a tensor parameter is available to the GPU and is tiled.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept DeviceRankTiledTensor = detail::IsDeviceRankTiledTensorV<Input, Rank, DataType>;
#endif

namespace detail {

template <typename T, typename... Args>
constexpr auto count_of_type(/*Args... args*/) {
    // return (std::is_same_v<Args, T> + ... + 0);
    return (std::is_convertible_v<Args, T> + ... + 0);
}

} // namespace detail

template <typename T, typename... Args>
concept NoneOfType = detail::count_of_type<T, Args...>() == 0;

template <typename T, typename... Args>
concept AtLeastOneOfType = detail::count_of_type<T, Args...>() >= 1;

template <typename T, size_t Num, typename... Args>
concept NumOfType = detail::count_of_type<T, Args...>() == Num;

} // namespace einsums