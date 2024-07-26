//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/*
 * Naming conventions:
 *
 * Each of the basic properties in TensorBases.hpp is converted into a test type.
 * If a basic property is called NameBase, then the test type is IsName.
 * The inline bool constant is then called IsNameV and the concept is called
 * NameConcept. The exception to this is that CoreTensorBase becomes IsIncoreTensor and
 * IsIncoreTensorV, but CoreTensorConcept.
 *
 * For combined types, it is a bit different. If the combined type is intended to completely
 * specify a tensor, then for a tensor type called Name, the test type, if it exists is called
 * IsRankName, the constant is IsRankNameV, and the concept is RankName. If the storage location
 * is specified, then the pattern is IsLocationRankName, IsLocationRankNameV, and LocationRankName.
 * If Location is Core, then it becomes IsIncoreRankName, IsIncoreRankNameV, and CoreRankName.
 * The templates for these will start with the type being tested, then the rank, then the stored
 * data type.
 *
 * For any other concept called Name, the test type, if it exists, will be IsName and the constant
 * will be IsNameV, converting any Core to Incore in the process.
 */

#pragma once

#include "einsums/utility/TensorBases.hpp"

#include <cstddef>
#include <type_traits>

namespace einsums {

namespace detail {

/**************************************
 *               Structs              *
 **************************************/

/**********************
 *    Basic traits.   *
 **********************/

/**
 * @struct IsTensor
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_props::TensorBase.
 *
 * @tparam D The type to check.
 */
template <typename D>
struct IsTensor : public std::is_base_of<tensor_props::TensorBaseNoExtra, D> {};

/**
 * @struct IsTypedTensor
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
struct IsTypedTensor : public std::is_base_of<tensor_props::TypedTensorBase<T>, D> {};

/**
 * @struct IsRankTensor
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 */
template <typename D, size_t Rank>
struct IsRankTensor : public std::is_base_of<tensor_props::RankTensorBase<Rank>, D> {};

/**
 * @struct IsLockableTensor
 *
 * @brief Tests whether the given tensor type can be locked.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct IsLockableTensor : public std::is_base_of<tensor_props::LockableTensorBase, D> {};

/**
 * @struct IsTRTensor
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * This checks to see if the tensor derives RankTensorBase and TypedTensorBase.
 * Try not to rely on a tensor deriving TRTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 */
template <typename D, size_t Rank, typename T>
struct IsTRTensor : std::bool_constant<IsRankTensor<D, Rank>::value && IsTypedTensor<D, T>::value> {};

/**
 * @struct IsTRLTensor
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * This checks to see if the tensor derives RankTensorBase, TypedTensorBase, and LockableTensorBase.
 * Try not to rely on a tensor deriving TRLTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 */
template <typename D, size_t Rank, typename T>
struct IsTRLTensor : std::bool_constant<IsRankTensor<D, Rank>::value && IsTypedTensor<D, T>::value && IsLockableTensor<D>::value> {};

/**
 * @struct IsIncoreTensor
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against CoreTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
struct IsIncoreTensor : public std::is_base_of<tensor_props::CoreTensorBase, D> {};

#ifdef __HIP__
/**
 * @struct IsDeviceTensor
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against DeviceTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
struct IsDeviceTensor : public std::is_base_of<tensor_props::DeviceTensorBase, D> {};
#endif

/**
 * @struct IsDiskTensor
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits DiskTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct IsDiskTensor : public std::is_base_of<tensor_props::DiskTensorBase, D> {};

/**
 * @struct IsTensorView
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks whether the tensor inherits TensorViewBaseNoExtra.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct IsTensorView : public std::is_base_of<tensor_props::TensorViewBaseNoExtra, D> {};

/**
 * @struct IsViewOf
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * Checks whether the tensor inherits the appropriate TensorViewBase.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 */
template <typename D, typename Viewed>
struct IsViewOf : public std::is_base_of<tensor_props::TensorViewBaseOnlyViewed<Viewed>, D> {};

/**
 * @struct IsBasicTensor
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the tensor inherits BasicTensorBaseNoExtra.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
struct IsBasicTensor : public std::is_base_of<tensor_props::BasicTensorBaseNoExtra, D> {};

/**
 * @struct IsCollectedTensor
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the tensor inherits CollectedTensorBaseOnlyStored if a type is given, or CollectedTensorBaseNoExtra if type is not
 * given.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
struct IsCollectedTensor : public std::bool_constant<std::is_void_v<StoredType>
                                                         ? std::is_base_of_v<tensor_props::CollectedTensorBaseNoExtra, D>
                                                         : std::is_base_of_v<tensor_props::CollectedTensorBaseOnlyStored<StoredType>, D>> {
};

/**
 * @struct IsTiledTensor
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
struct IsTiledTensor : public std::bool_constant<std::is_base_of_v<tensor_props::TiledTensorBaseNoExtra, D> &&
                                                 (std::is_void_v<StoredType> ||
                                                  std::is_base_of_v<tensor_props::CollectedTensorBaseOnlyStored<StoredType>, D>)> {};

/**
 * @struct IsBlockTensor
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
struct IsBlockTensor : public std::bool_constant<std::is_base_of_v<tensor_props::BlockTensorBaseNoExtra, D> &&
                                                 (std::is_void_v<StoredType> ||
                                                  std::is_base_of_v<tensor_props::CollectedTensorBaseOnlyStored<StoredType>, D>)> {};

/**
 * @struct IsFunctionTensor
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct IsFunctionTensor : public std::is_base_of<tensor_props::FunctionTensorBaseNoExtra, D> {};

/********************************
 *      Inline definitions      *
 ********************************/

/**
 * @property IsTensorV
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_props::TensorBase.
 *
 * @tparam D The type to check.
 */
template <typename D>
constexpr inline bool IsTensorV = IsTensor<D>::value;

/**
 * @property IsTypedTensorV
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
constexpr inline bool IsTypedTensorV = IsTypedTensor<D, T>::value;

/**
 * @property IsRankTensorV
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 */
template <typename D, size_t Rank>
constexpr inline bool IsRankTensorV = IsRankTensor<D, Rank>::value;

/**
 * @property IsLockableTensorV
 *
 * @brief Tests whether the given tensor type can be locked.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsLockableTensorV = IsLockableTensor<D>::value;

/**
 * @property IsTRTensorV
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * This checks to see if the tensor derives RankTensorBase and TypedTensorBase.
 * Try not to rely on a tensor deriving TRTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsTRTensorV = IsTRTensor<D, Rank, T>::value;

/**
 * @property IsTRLTensorV
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * This checks to see if the tensor derives RankTensorBase, TypedTensorBase, and LockableTensorBase.
 * Try not to rely on a tensor deriving TRLTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsTRLTensorV = IsTRLTensor<D, Rank, T>::value;

/**
 * @property IsIncoreTensorV
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against CoreTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsIncoreTensorV = IsIncoreTensor<D>::value;

#ifdef __HIP__
/**
 * @property IsDeviceTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against DeviceTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsDeviceTensorV = IsDeviceTensor<D>::value;
#endif

/**
 * @property IsDiskTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits DiskTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsDiskTensorV = IsDiskTensor<D>::value;

/**
 * @property IsTensorViewV
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks whether the tensor inherits TensorViewBaseNoExtra.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsTensorViewV = IsTensorView<D>::value;

/**
 * @property IsViewOfV
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * Checks whether the tensor inherits the appropriate TensorViewBase.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 */
template <typename D, typename Viewed>
constexpr inline bool IsViewOfV = IsViewOf<D, Viewed>::value;

/**
 * @property IsBasicTensorV
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the tensor inherits BasicTensorBaseNoExtra.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsBasicTensorV = IsBasicTensor<D>::value;

/**
 * @property IsCollectedTensorV
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the tensor inherits CollectedTensorBaseOnlyStored if a type is given, or CollectedTensorBaseNoExtra if type is not
 * given.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
constexpr inline bool IsCollectedTensorV = IsCollectedTensor<D, StoredType>::value;

/**
 * @property IsTiledTensorV
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
constexpr inline bool IsTiledTensorV = IsTiledTensor<D, StoredType>::value;

/**
 * @property IsBlockTensorV
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
constexpr inline bool IsBlockTensorV = IsBlockTensor<D, StoredType>::value;

/**
 * @property IsFunctionTensorV
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsFunctionTensorV = IsFunctionTensor<D>::value;

/**************************************
 *        Combined expressions        *
 **************************************/

/**
 * @property IsSamePlaceV
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 */
template <typename... Tensors>
#ifdef __HIP__
constexpr inline bool IsInSamePlaceV =
    (IsIncoreTensorV<Tensors> && ...) || (IsDeviceTensorV<Tensors> && ...) || (IsDiskTensorV<Tensors> && ...);
#else
constexpr inline bool IsInSamePlaceV = (IsIncoreTensorV<Tensors> && ...) || (IsDiskTensorV<Tensors> && ...);
#endif

/**
 * @property IsIncoreRankTensorV
 *
 * @brief Requires that a tensor is in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankTensorV = IsIncoreTensorV<D> && IsTRTensorV<D, Rank, T>;

#ifdef __HIP__
/**
 * @property IsDeviceRankTensorV
 *
 * @brief Requires that a tensor is available to the graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankTensorV = IsDeviceTensorV<D> && IsTRTensorV<D, Rank, T>;
#endif

/**
 * @property IsDiskRankTensorV
 *
 * @brief Requires that a tensor is stored on disk, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDiskRankTensorV = IsDiskTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsRankBasicTensorV
 *
 * @brief Requires that a tensor is a basic tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsRankBasicTensorV = IsBasicTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsRankTiledTensorV
 *
 * @brief Requires that a tensor is a Tiled tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsRankTiledTensorV = IsTiledTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsRankBlockTensorV
 *
 * @brief Requires that a tensor is a block tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsRankBlockTensorV = IsBlockTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsIncoreRankBasicTensorV
 *
 * @brief Requires that a tensor is a basic tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankBasicTensorV = IsBasicTensorV<D> && IsTRTensorV<D, Rank, T> && IsIncoreTensorV<D>;

#ifdef __HIP__
/**
 * @property IsDeviceRankBasicTensorV
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankBasicTensorV = IsBasicTensorV<D> && IsTRTensorV<D, Rank, T> && IsDeviceTensorV<D>;
#endif

/**
 * @property IsIncoreRankBlockTensorV
 *
 * @brief Requires that a tensor is a block tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankBlockTensorV = IsBlockTensorV<D> && IsTRTensorV<D, Rank, T> && IsIncoreTensorV<D>;

#ifdef __HIP__
/**
 * @property IsDeviceRankBlockTensorV
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankBlockTensorV = IsBlockTensorV<D> && IsTRTensorV<D, Rank, T> && IsDeviceTensorV<D>;
#endif

/**
 * @property IsIncoreRankTiledTensorV
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankTiledTensorV = IsTiledTensorV<D> && IsTRTensorV<D, Rank, T> && IsIncoreTensorV<D>;

#ifdef __HIP__
/**
 * @property IsDeviceRankTiledTensorV
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankTiledTensorV = IsTiledTensorV<D> && IsTRTensorV<D, Rank, T> && IsDeviceTensorV<D>;
#endif

/**
 * @property IsIncoreBasicTensorV
 *
 * @brief Checks to see if the tensor is available in-core and is a basic tensor.
 *
 * Checks the tensor against CoreTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsIncoreBasicTensorV = IsIncoreTensor<D>::value && IsBasicTensor<D>::value;

#ifdef __HIP__
/**
 * @property IsDeviceBasicTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a basic tensor.
 *
 * Checks the tensor against DeviceTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsDeviceBasicTensorV = IsDeviceTensor<D>::value && IsBasicTensor<D>::value;
#endif

/**
 * @property IsDiskBasicTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk and is a basic tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsDiskBasicTensorV = IsDiskTensor<D>::value && IsBasicTensor<D>::value;

/**
 * @property IsIncoreTiledTensorV
 *
 * @brief Checks to see if the tensor is available in-core and is a basic tensor.
 *
 * Checks the tensor against CoreTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsIncoreTiledTensorV = IsIncoreTensor<D>::value && IsTiledTensor<D>::value;

#ifdef __HIP__
/**
 * @property IsDeviceTiledTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a tiled tensor.
 *
 * Checks the tensor against DeviceTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsDeviceTiledTensorV = IsDeviceTensor<D>::value && IsTiledTensor<D>::value;
#endif

/**
 * @property IsDiskTiledTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk and is a tiled tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsDiskTiledTensorV = IsDiskTensor<D>::value && IsTiledTensor<D>::value;

/**
 * @property IsIncoreBlockTensorV
 *
 * @brief Checks to see if the tensor is available in-core and is a block tensor.
 *
 * Checks the tensor against CoreTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsIncoreBlockTensorV = IsIncoreTensor<D>::value && IsBlockTensor<D>::value;

#ifdef __HIP__
/**
 * @property IsDeviceBlockTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a block tensor.
 *
 * Checks the tensor against DeviceTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsDeviceBlockTensorV = IsDeviceTensor<D>::value && IsBlockTensor<D>::value;
#endif

/**
 * @property IsDiskBlockTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk and is a block tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsDiskBlockTensorV = IsDiskTensor<D>::value && IsBlockTensor<D>::value;

/**
 * @property IsSameUnderlyingV
 *
 * @brief Checks to see if the tensors have the same storage type, but without specifying that type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameUnderlyingV = (std::is_same_v<typename First::data_type, typename Rest::data_type> && ...);

/**
 * @property IsSameRankV
 *
 * @brief Checks to see if the tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameRankV = ((First::rank == Rest::rank) && ...);

/**
 * @property IsSameUnderlyingAndRankV
 *
 * @brief Checks to see if the tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameUnderlyingAndRankV = IsSameUnderlyingV<First, Rest...> && IsSameRankV<First, Rest...>;

} // namespace detail

/**
 * @concept TensorConcept
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_props::TensorBase.
 *
 * @tparam D The type to check.
 */
template <typename D>
concept TensorConcept = detail::IsTensorV<D>;

/**
 * @concept TypedTensorConcept
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
concept TypedTensorConcept = detail::IsTypedTensorV<D, T>;

/**
 * @concept RankTensorConcept
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 */
template <typename D, size_t Rank>
concept RankTensorConcept = detail::IsRankTensorV<D, Rank>;

/**
 * @concept LockableTensorConcept
 *
 * @brief Tests whether the given tensor type can be locked.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept LockableTensorConcept = detail::IsLockableTensorV<D>;

/**
 * @concept TRTensorConcept
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * This checks to see if the tensor derives RankTensorBase and TypedTensorBase.
 * Try not to rely on a tensor deriving TRTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 */
template <typename D, size_t Rank, typename T>
concept TRTensorConcept = detail::IsTRTensorV<D, Rank, T>;

/**
 * @concept TRLTensorConcept
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * This checks to see if the tensor derives RankTensorBase, TypedTensorBase, and LockableTensorBase.
 * Try not to rely on a tensor deriving TRLTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 */
template <typename D, size_t Rank, typename T>
concept TRLTensorConcept = detail::IsTRLTensorV<D, Rank, T>;

/**
 * @concept CoreTensorConcept
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against CoreTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreTensorConcept = detail::IsIncoreTensorV<D>;

#ifdef __HIP__
/**
 * @concept DeviceTensorConcept
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against DeviceTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceTensorConcept = detail::IsDeviceTensorV<D>;
#endif

/**
 * @concept DiskTensorConcept
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits DiskTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept DiskTensorConcept = detail::IsDiskTensorV<D>;

/**
 * @concept TensorViewConcept
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks whether the tensor inherits TensorViewBaseNoExtra.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept TensorViewConcept = detail::IsTensorViewV<D>;

/**
 * @concept ViewOfConcept
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * Checks whether the tensor inherits the appropriate TensorViewBase.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 */
template <typename D, typename Viewed>
concept ViewOfConcept = detail::IsViewOfV<D, Viewed>;

/**
 * @concept BasicTensorConcept
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the tensor inherits BasicTensorBaseNoExtra.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept BasicTensorConcept = detail::IsBasicTensorV<D>;

/**
 * @concept CollectedTensorConcept
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the tensor inherits CollectedTensorBaseOnlyStored if a type is given, or CollectedTensorBaseNoExtra if type is not
 * given.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
concept CollectedTensorConcept = detail::IsCollectedTensorV<D, StoredType>;

/**
 * @concept TiledTensorConcept
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
concept TiledTensorConcept = detail::IsTiledTensorV<D, StoredType>;

/**
 * @concept BlockTensorConcept
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
concept BlockTensorConcept = detail::IsBlockTensorV<D, StoredType>;

/**
 * @concept FunctionTensorConcept
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept FunctionTensorConcept = detail::IsFunctionTensorV<D>;

/**
 * @concept CoreRankTensor
 *
 * @brief Requires that a tensor is in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankTensor = detail::IsIncoreRankTensorV<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankTensor
 *
 * @brief Requires that a tensor is available to the graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankTensor = detail::IsDeviceRankTensorV<D, Rank, T>;
#endif

/**
 * @concept DiskRankTensor
 *
 * @brief Requires that a tensor is stored on disk, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DiskRankTensor = detail::IsDiskRankTensorV<D, Rank, T>;

/**
 * @concept RankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept RankBasicTensor = detail::IsRankBasicTensorV<D, Rank, T>;

/**
 * @concept RankTiledTensor
 *
 * @brief Requires that a tensor is a Tiled tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept RankTiledTensor = detail::IsRankTiledTensorV<D, Rank, T>;

/**
 * @concept RankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept RankBlockTensor = detail::IsRankBlockTensorV<D, Rank, T>;

/**
 * @concept CoreRankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankBasicTensor = detail::IsIncoreRankBasicTensorV<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankBasicTensor = detail::IsDeviceRankBasicTensorV<D, Rank, T>;
#endif

/**
 * @concept CoreRankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankBlockTensor = detail::IsIncoreRankBlockTensorV<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankBlockTensor = detail::IsDeviceRankBlockTensorV<D, Rank, T>;
#endif

/**
 * @concept CoreRankTiledTensor
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankTiledTensor = detail::IsIncoreRankTiledTensorV<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankTiledTensor
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankTiledTensor = detail::IsDeviceRankTiledTensorV<D, Rank, T>;
#endif

/**
 * @concept CoreBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor stored in-core.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D>
concept CoreBasicTensorConcept = detail::IsIncoreBasicTensorV<D>;

#ifdef __HIP__
/**
 * @concept DeviceBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceBasicTensorConcept = detail::IsDeviceBasicTensorV<D>;
#endif

/**
 * @concept CoreBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreBlockTensorConcept = detail::IsIncoreBlockTensorV<D>;

#ifdef __HIP__
/**
 * @concept DeviceBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceBlockTensorConcept = detail::IsDeviceBlockTensorV<D>;
#endif

/**
 * @concept CoreTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreTiledTensorConcept = detail::IsIncoreTiledTensorV<D>;

#ifdef __HIP__
/**
 * @concept DeviceTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceTiledTensorConcept = detail::IsDeviceTiledTensorV<D>;
#endif

/**
 * @concept InSamePlace
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 */
template <typename... Tensors>
concept InSamePlace = detail::IsInSamePlaceV<Tensors...>;

/**
 * @concept MatrixConcept
 *
 * @brief Alias of RankTensorConcept<D, 2>.
 *
 * Shorthand for requiring that a tensor be a matrix.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept MatrixConcept = RankTensorConcept<D, 2>;

/**
 * @concept VectorConcept
 *
 * @brief Alias of RankTensorConcept<D, 1>.
 *
 * Shorthand for requiring that a tensor be a vector.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept VectorConcept = RankTensorConcept<D, 1>;

/**
 * @concept ScalarConcept
 *
 * @brief Alias of RankTensorConcept<D, 0>.
 *
 * Shorthand for requiring that a tensor be a scalar.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept ScalarConcept = RankTensorConcept<D, 0>;

/**
 * @concept SameUnderlying
 *
 * @brief Checks that several tensors store the same type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameUnderlying = detail::IsSameUnderlyingV<First, Rest...>;

/**
 * @concept SameRank
 *
 * @brief Checks that several tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameRank = detail::IsSameRankV<First, Rest...>;

/**
 * @concept SameUnderlyingAndRank
 *
 * @brief Checks that several tensors have the same rank and underlying type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameUnderlyingAndRank = detail::IsSameUnderlyingAndRankV<First, Rest...>;

/**
 * @struct remove_view
 *
 * @brief Gets the underlying type of a view.
 *
 * @tparam D The tensor type to strip.
 */
template <typename D>
struct remove_view {
  public:
    using base_type = D;
};

template <TensorViewConcept D>
struct remove_view<D> {
  public:
    using base_type = typename D::underlying_type;
};

/**
 * @typedef remove_view_t
 *
 * @brief Gets the underlying type of a view.
 *
 * @tparam D The tensor type to strip.
 */
template <typename D>
using remove_view_t = typename remove_view<D>::base_type;

namespace detail {
/**
 * @brief Creates a new tensor with the same type as the input but with a different rank or storage type.
 *
 * This does not initialize the new tensor and more or less is used to get the return type with a decltype.
 *
 * @param tensor The tensor whose type is being copied.
 * @tparam TensorType The type of the
 */
template <typename NewT, size_t NewRank, template <typename, size_t> typename TensorType, typename T, size_t Rank>
    requires(TensorConcept<TensorType<T, Rank>>)
TensorType<NewT, NewRank> create_tensor_of_same_type(const TensorType<T, Rank> &tensor) {
    return TensorType<NewT, NewRank>();
}

/**
 * @brief Creates a new basic tensor in the same place as the input, but with a different rank and storage type..
 *
 * This does not initialize the new tensor and more or less is used to get the return type with a decltype.
 *
 * @param tensor The tensor whose type is being copied.
 * @tparam TensorType The type of the
 */
template <typename NewT, size_t NewRank, CoreTensorConcept TensorType>
Tensor<NewT, NewRank> create_basic_tensor_like(const TensorType &tensor) {
    return Tensor<NewT, NewRank>();
}

#ifdef __HIP__
template <typename NewT, size_t NewRank, DeviceTensorConcept TensorType>
DeviceTensor<NewT, NewRank> create_basic_tensor_like(const TensorType &tensor) {
    return DeviceTensor<NewT, NewRank>();
}
#endif

template <typename NewT, size_t NewRank, DiskTensorConcept TensorType>
DiskTensor<NewT, NewRank> create_basic_tensor_like(const TensorType &tensor) {
    return DiskTensor<NewT, NewRank>();
}

} // namespace detail

/**
 * @typedef TensorLike
 *
 * @brief Gets the type of a tensor, but with a new rank and type.
 *
 * @tparam D The underlying tensor type.
 * @tparam T The new type.
 * @tparam Rank The new rank.
 */
template <TensorConcept D, typename T, size_t Rank>
using TensorLike = decltype(detail::create_tensor_of_same_type<T, Rank>(D()));

/**
 * @typedef BasicTensorLike
 *
 * @brief Gets the type of basic tensor with the same storage location, but with different rank and underlying type.
 *
 * @tparam D The underlying tensor type.
 * @tparam T The new type.
 * @tparam Rank The new rank.
 */
template <TensorConcept D, typename T, size_t Rank>
using BasicTensorLike = decltype(detail::create_basic_tensor_like<T, Rank>(D()));

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