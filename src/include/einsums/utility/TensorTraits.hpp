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
#endif

namespace detail {

template <typename D, size_t Rank, typename T>
struct IsIncoreRankTensor
    : public std::bool_constant<std::is_same_v<std::decay_t<D>, Tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, TensorView<T, Rank>> ||
                                std::is_same_v<std::decay_t<D>, BlockTensor<T, Rank>>> {};
template <typename D, size_t Rank, typename T>
inline constexpr bool IsIncoreRankTensorV = IsIncoreRankTensor<D, Rank, T>::value;

template <typename D, size_t Rank, typename T>
struct IsIncoreRankBlockTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, BlockTensor<T, Rank>>> {};

template<typename D, size_t Rank, typename T>
inline constexpr bool IsIncoreRankBlockTensorV = IsIncoreRankBlockTensor<D, Rank, T>::value;

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
 * @struct IsDeviceRankBlockTensor
 *
 * @brief Struct for specifying that a tensor is device compatible, and is block diagonal.
 */
template <typename D, size_t Rank, typename T>
struct IsDeviceRankBlockTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, BlockDeviceTensor<T, Rank>>> {};

/**
 * @property IsDeviceRankTensorV
 *
 * @brief True if the tensor is device compatible.
 */
template <typename D, size_t Rank, typename T>
inline constexpr bool IsDeviceRankTensorV = IsDeviceRankTensor<D, Rank, T>::value;

/**
 * @property IsDeviceRankTensorV
 *
 * @brief True if the tensor is device compatible and is block diagonal.
 */
template <typename D, size_t Rank, typename T>
inline constexpr bool IsDeviceRankBlockTensorV = IsDeviceRankBlockTensor<D, Rank, T>::value;
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

template<typename Input, size_t Rank, typename DataType = double>
concept CoreRankBlockTensor = detail::IsIncoreRankBlockTensorV<Input, Rank, DataType>;

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
 * @concept DeviceRankBlockTensor
 *
 * @brief Concept for testing whether a tensor parameter is available to the GPU and is block diagonal.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept DeviceRankBlockTensor = detail::IsDeviceRankBlockTensorV<Input, Rank, DataType>;
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