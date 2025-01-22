//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TensorBase/TensorBase.hpp>

#include <cstddef>
#include <type_traits>

namespace einsums {

// Forward declarations
template <typename T, size_t Rank>
struct Tensor;

namespace disk {
template <typename T, size_t Rank>
struct Tensor;
}

#if defined(EINSUMS_COMPUTE_CODE)
template <typename T, size_t Rank>
struct DeviceTensor;
#endif

/********************************
 *      Inline definitions      *
 ********************************/

/**
 * @property IsTensorV
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type defines the functions full_view_of_underlying, name, dim, and dims.
 *
 * @tparam D The type to check.
 */
template <typename D>
constexpr inline bool IsTensorV = requires(D tensor) {
    { tensor.full_view_of_underlying() } -> std::convertible_to<bool>;
    { tensor.name() } -> std::convertible_to<std::string>;
    tensor.dim(std::declval<int>());
    tensor.dims();
    typename D::ValueType;
};

/**
 * @property IsTypedTensorV
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
constexpr inline bool IsTypedTensorV = requires {
    typename D::ValueType;
    requires std::is_same_v<typename D::ValueType, T>;
};

/**
 * @property IsRankTensorV
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have. If the rank is -1, it only checks that the tensor has a rank that is known at compile time.
 */
template <typename D, ptrdiff_t Rank = -1>
constexpr inline bool IsRankTensorV = requires {
    D::Rank;
    requires(Rank == -1) || (Rank == D::Rank);
};

/**
 * @param IsScalarV
 *
 * @brief Tests to see if a value is a scalar value.
 *
 * Checks to see if a type is either a tensor with rank 0 or a scalar type such as double or complex<float>.
 *
 * @tparam D The type to check.
 */
template <typename D>
constexpr inline bool IsScalarV = IsRankTensorV<D, 0> || !IsTensorV<D>;

/**
 * @property IsBasicLockableV
 *
 * @brief Tests whether a given type satisfies C++'s BasicLockable requirement.
 */
template <typename D>
constexpr inline bool IsBasicLockableV = requires(D var) {
    var.lock();
    var.unlock();
};

/**
 * @property IsLockableV
 *
 * @brief Tests whether the given type satisfies C++'s Lockable requirement.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsLockableV = requires(D var) {
    requires IsBasicLockableV<D>;
    { var.try_lock() } -> std::same_as<bool>;
};

/**
 * @property IsTRTensorV
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsTRTensorV = IsTypedTensorV<D, T> && IsRankTensorV<D, Rank>;

/**
 * @property IsTRLTensorV
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsTRLTensorV = IsTypedTensorV<D, T> && IsRankTensorV<D, Rank> && IsBasicLockableV<D>;

/**
 * @property IsIncoreTensorV
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against tensor_base::CoreTensor.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsIncoreTensorV = std::is_base_of_v<einsums::tensor_base::CoreTensor, D>;

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @property IsDeviceTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against tensor_base::DeviceTensor.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsDeviceTensorV = std::is_base_of_v<einsums::tensor_base::DeviceTensorBase, D>;
#endif

/**
 * @property IsDiskTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits tensor_base::DiskTensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsDiskTensorV = std::is_base_of_v<einsums::tensor_base::DiskTensor, D>;

/**
 * @property IsTensorViewV
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks to see if the type has a typedef called @code underlying_type .
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsTensorViewV = requires { typename D::underlying_type; };

/**
 * @property IsViewOfV
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 */
template <typename D, typename Viewed>
constexpr inline bool IsViewOfV = requires {
    requires IsTensorViewV<D>;
    requires std::is_same_v<typename D::underlying_type, Viewed>;
};

/**
 * @property IsBasicTensorV
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the type defines the functions data, stride, and strides.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsBasicTensorV = requires(D tensor) {
    tensor.data();
    tensor.stride(0);
    tensor.strides();
};

/**
 * @property IsCollectedTensorV
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the type defines the type @code StoredType .
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename stored_type = void>
constexpr inline bool IsCollectedTensorV = requires {
    typename D::StoredType;
    requires std::is_same_v<stored_type, void> || std::is_same_v<stored_type, typename D::StoredType>;
};

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
constexpr inline bool IsTiledTensorV = requires {
    requires std::is_base_of_v<einsums::tensor_base::TiledTensorNoExtra, D>;
    requires std::is_same_v<StoredType, void> || std::is_same_v<typename D::StoredType, StoredType>;
};

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
constexpr inline bool IsBlockTensorV = requires {
    requires std::is_base_of_v<einsums::tensor_base::BlockTensorNoExtra, D>;
    requires std::is_same_v<StoredType, void> || std::is_same_v<typename D::StoredType, StoredType>;
};

namespace detail {

template <size_t index>
int get_zero() {
    return 0;
}

template <typename T, size_t... ints>
constexpr bool test_application(std::index_sequence<ints...> const &) {
    return false;
}

template <typename T, size_t... ints>
    requires requires(T tensor, std::index_sequence<ints...> seq) { tensor(get_zero<ints>()...); }
constexpr bool test_application(std::index_sequence<ints...> const &) {
    return true;
}
} // namespace detail

/**
 * @property IsFunctionTensorV
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * More specifically, checks to see if the tensor can be indexed using
 * function call syntax.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsFunctionTensorV = requires {
    requires detail::test_application<D>(std::make_index_sequence<D::Rank>());
    requires IsRankTensorV<D>;
};

/**
 * @property IsAlgebraTensorV
 *
 * @brief Checks to see if operations with the tensor can be optimized using libraries, indicated by deriving AlgebraOptimizedTensor.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsAlgebraTensorV = std::is_base_of_v<einsums::tensor_base::AlgebraOptimizedTensor, D>;

/**************************************
 *        Combined expressions        *
 **************************************/

/**
 * @property IsInSamePlaceV
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 */
template <typename... Tensors>
constexpr inline bool IsInSamePlaceV = (IsIncoreTensorV<Tensors> && ...) || (IsDiskTensorV<Tensors> && ...)
#if defined(EINSUMS_COMPUTE_CODE)
                                       || (IsDeviceTensorV<Tensors> && ...)
#endif
    ;

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

#if defined(EINSUMS_COMPUTE_CODE)
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

#if defined(EINSUMS_COMPUTE_CODE)
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

#if defined(EINSUMS_COMPUTE_CODE)
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

#if defined(EINSUMS_COMPUTE_CODE)
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
constexpr inline bool IsIncoreBasicTensorV = IsIncoreTensorV<D> && IsBasicTensorV<D>;

#if defined(EINSUMS_COMPUTE_CODE)
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
constexpr inline bool IsDeviceBasicTensorV = IsDeviceTensorV<D> && IsBasicTensorV<D>;
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
constexpr inline bool IsDiskBasicTensorV = IsDiskTensorV<D> && IsBasicTensorV<D>;

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
constexpr inline bool IsIncoreTiledTensorV = IsIncoreTensorV<D> && IsTiledTensorV<D>;

#if defined(EINSUMS_COMPUTE_CODE)
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
constexpr inline bool IsDeviceTiledTensorV = IsDeviceTensorV<D> && IsTiledTensorV<D>;
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
constexpr inline bool IsDiskTiledTensorV = IsDiskTensorV<D> && IsTiledTensorV<D>;

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
constexpr inline bool IsIncoreBlockTensorV = IsIncoreTensorV<D> && IsBlockTensorV<D>;

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @property IsDeviceBLockTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a block tensor.
 *
 * Checks the tensor against DeviceTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool IsDeviceBLockTensorV = IsDeviceTensorV<D> && IsBlockTensorV<D>;
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
constexpr inline bool IsDiskBlockTensorV = IsDiskTensorV<D> && IsBlockTensorV<D>;

/**
 * @property IsSameUnderlyingV
 *
 * @brief Checks to see if the tensors have the same storage type, but without specifying that type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameUnderlyingV = (std::is_same_v<typename First::ValueType, typename Rest::ValueType> && ...);

/**
 * @property IsSameRankV
 *
 * @brief Checks to see if the tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameRankV = ((First::Rank == Rest::Rank) && ...);

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
concept TensorConcept = IsTensorV<D>;

template <typename D>
concept NotTensorConcept = !IsTensorV<D>;

/**
 * @concept TypedTensorConcept
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
concept TypedTensorConcept = IsTypedTensorV<D, T>;

/**
 * @concept RankTensorConcept
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 */
template <typename D, ptrdiff_t Rank = -1>
concept RankTensorConcept = IsRankTensorV<D, Rank>;

/**
 * @concept BasicLockableConcept
 *
 * @brief Tests whether the given type statisfies the C++ BasicLockable requirement.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept BasicLockableConcept = IsBasicLockableV<D>;

/**
 * @concept LockableConcept
 *
 * @brief Tests whether the given type satisfies the C++ Lockable requirement.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept LockableConcept = IsLockableV<D>;

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
concept TRTensorConcept = IsTRTensorV<D, Rank, T>;

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
concept TRLTensorConcept = IsTRLTensorV<D, Rank, T>;

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
concept CoreTensorConcept = IsIncoreTensorV<D>;

#if defined(EINSUMS_COMPUTE_CODE)
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
concept DeviceTensorConcept = IsDeviceTensorV<D>;
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
concept DiskTensorConcept = IsDiskTensorV<D>;

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
concept TensorViewConcept = IsTensorViewV<D>;

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
concept ViewOfConcept = IsViewOfV<D, Viewed>;

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
concept BasicTensorConcept = IsBasicTensorV<D>;

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
concept CollectedTensorConcept = IsCollectedTensorV<D, StoredType>;

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
concept TiledTensorConcept = IsTiledTensorV<D, StoredType>;

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
concept BlockTensorConcept = IsBlockTensorV<D, StoredType>;

/**
 * @concept FunctionTensorConcept
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept FunctionTensorConcept = IsFunctionTensorV<D>;

/**
 * @concept AlgebraTensorConcept
 *
 * @brief Checks to see if operations with the tensor can be optimized with libraries.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept AlgebraTensorConcept = IsAlgebraTensorV<D>;

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
concept CoreRankTensor = IsIncoreRankTensorV<D, Rank, T>;

#if defined(EINSUMS_COMPUTE_CODE)
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
concept DeviceRankTensor = IsDeviceRankTensorV<D, Rank, T>;
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
concept DiskRankTensor = IsDiskRankTensorV<D, Rank, T>;

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
concept RankBasicTensor = IsRankBasicTensorV<D, Rank, T>;

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
concept RankTiledTensor = IsRankTiledTensorV<D, Rank, T>;

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
concept RankBlockTensor = IsRankBlockTensorV<D, Rank, T>;

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
concept CoreRankBasicTensor = IsIncoreRankBasicTensorV<D, Rank, T>;

#if defined(EINSUMS_COMPUTE_CODE)
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
concept DeviceRankBasicTensor = IsDeviceRankBasicTensorV<D, Rank, T>;
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
concept CoreRankBlockTensor = IsIncoreRankBlockTensorV<D, Rank, T>;

#if defined(EINSUMS_COMPUTE_CODE)
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
concept DeviceRankBlockTensor = IsDeviceRankBlockTensorV<D, Rank, T>;
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
concept CoreRankTiledTensor = IsIncoreRankTiledTensorV<D, Rank, T>;

#if defined(EINSUMS_COMPUTE_CODE)
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
concept DeviceRankTiledTensor = IsDeviceRankTiledTensorV<D, Rank, T>;
#endif

/**
 * @concept CoreBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreBasicTensorConcept = IsIncoreBasicTensorV<D>;

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @concept DeviceBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceBasicTensorConcept = IsDeviceBasicTensorV<D>;
#endif

/**
 * @concept CoreBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreBlockTensorConcept = IsIncoreBlockTensorV<D>;

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @concept DeviceBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceBlockTensorConcept = IsDeviceBLockTensorV<D>;
#endif

/**
 * @concept CoreTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreTiledTensorConcept = IsIncoreTiledTensorV<D>;

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @concept DeviceTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceTiledTensorConcept = IsDeviceTiledTensorV<D>;
#endif

/**
 * @concept InSamePlace
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 */
template <typename... Tensors>
concept InSamePlace = IsInSamePlaceV<Tensors...>;

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
 * Shorthand for requiring that a tensor be a scalar. That is, a tensor with zero rank or a variable with a type such as double or
 * std::complex<float>.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept ScalarConcept = IsScalarV<D>;

/**
 * @concept SameUnderlying
 *
 * @brief Checks that several tensors store the same type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameUnderlying = IsSameUnderlyingV<First, Rest...>;

/**
 * @concept SameRank
 *
 * @brief Checks that several tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameRank = IsSameRankV<First, Rest...>;

/**
 * @concept SameUnderlyingAndRank
 *
 * @brief Checks that several tensors have the same rank and underlying type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameUnderlyingAndRank = IsSameUnderlyingAndRankV<First, Rest...>;

/**
 * @struct RemoveView
 *
 * @brief Gets the underlying type of view.
 *
 * @tparam D The tensor type to strip.
 */
template <typename D>
struct RemoveView {
    /**
     * @typedef base_type
     *
     * This will contain the type of tensor being viewed by the tensor view, if given a view.
     * Otherwise, it will be the same as the tensor being passed.
     */
    using base_type = D;
};

#ifndef DOXYGEN
template <TensorViewConcept D>
struct RemoveView<D> {
    using base_type = typename D::underlying_type;
};
#endif

/**
 * @typedef RemoveViewT
 *
 * @brief Gets the underlying type of a view.
 *
 * @tparam D The tensor type to strip.
 */
template <typename D>
using RemoveViewT = typename RemoveView<D>::base_type;

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
TensorType<NewT, NewRank> create_tensor_of_same_type(TensorType<T, Rank> const &tensor) {
    return TensorType<NewT, NewRank>();
}

/**
 * @brief Creates a new basic tensor in the same place as the input, but with a different rank and storage type.
 *
 * This does not initialize the new tensor and more or less is used to get the return type with a decltype.
 *
 * @param tensor The tensor whose type is being copied.
 * @tparam TensorType The type of the tensor to be copied.
 */
template <typename NewT, size_t NewRank, CoreTensorConcept TensorType>
Tensor<NewT, NewRank> create_basic_tensor_like(TensorType const &tensor) {
    return Tensor<NewT, NewRank>();
}

#ifndef DOXYGEN
#    if defined(EINSUMS_COMPUTE_CODE)
template <typename NewT, size_t NewRank, DeviceTensorConcept TensorType>
DeviceTensor<NewT, NewRank> create_basic_tensor_like(TensorType const &tensor) {
    return DeviceTensor<NewT, NewRank>();
}
#    endif

template <typename NewT, size_t NewRank, DiskTensorConcept TensorType>
disk::Tensor<NewT, NewRank> create_basic_tensor_like(TensorType const &) {
    return disk::Tensor<NewT, NewRank>();
}
#endif

} // namespace detail

/**
 * @typedef TensorLike
 *
 * @brief Gets the type of tensor, but with a new rank and type.
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

/**
 * @struct ValueType
 *
 * @brief Gets the data type of tensor/scalar.
 *
 * Normally, you can get the data type using an expression such as typename AType::ValueType. However, if you want
 * to support both zero-rank tensors and scalars, then this typedef can help with brevity.
 */
template <typename D>
struct ValueType {
    /**
     * @typedef type
     *
     * This will contain the type of data stored by the tensor.
     */
    using type = D;
};

#ifndef DOXYGEN
template <TensorConcept D>
struct ValueType<D> {
    using type = typename D::ValueType;
};
#endif

/**
 * @typedef ValueTypeT
 *
 * @brief Gets the data type of tensor/scalar.
 *
 * Normally, you can get the data type using an expression such as typename AType::ValueType. However, if you want
 * to support both zero-rank tensors and scalars, then this typedef can help with brevity.
 */
template <typename D>
using ValueTypeT = typename ValueType<D>::type;

/**
 * @property TensorRank
 *
 * @brief Gets the rank of a tensor/scalar.
 *
 * Normally, you can get the rank using an expression such as AType::Rank. However,
 * if you want to support both zero-rank tensors and scalars, then this constant can help with brevity.
 */
template <typename D>
constexpr size_t TensorRank = 0;

template <TensorConcept D>
constexpr size_t TensorRank<D> = D::Rank;

/**
 * @struct BiggestType
 *
 * @brief Gets the type with the biggest storage specification.
 */
template <typename First, typename... Rest>
struct BiggestType {
    /**
     * @typedef type
     *
     * The result of the operation. This will be the biggest type passed to this struct.
     */
    using type =
        std::conditional_t<(sizeof(First) > sizeof(typename BiggestType<Rest...>::type)), First, typename BiggestType<Rest...>::type>;
};

#ifndef DOXYGEN
template <typename First>
struct BiggestType<First> {
    using type = First;
};
#endif

/**
 * @typedef BiggestTypeT
 *
 * @brief Gets the type with the biggest storage specification.
 */
template <typename... Args>
using BiggestTypeT = typename BiggestType<Args...>::type;

/**
 * @struct LocationTensorBaseOf
 *
 * @brief Gets the location base (CoreTensorBase, DiskTensorBase, etc.) of the argument.
 *
 * @tparam D The tensor type to query.
 */
template <typename D>
struct LocationTensorBaseOf {};

#ifndef DOXYGEN
template <CoreTensorConcept D>
struct LocationTensorBaseOf<D> {
    using type = tensor_base::CoreTensor;
};

template <DiskTensorConcept D>
struct LocationTensorBaseOf<D> {
    using type = tensor_base::DiskTensor;
};

#    if defined(EINSUMS_COMPUTE_CODE)
template <DeviceTensorConcept D>
struct LocationTensorBaseOf<D> {
    using type = tensor_base::DeviceTensorBase;
};
#    endif
#endif

/**
 * @typedef LocationTensorBaseOfT
 *
 * @brief Gets the location base (CoreTensorBase, DiskTensorBase, etc.) of the argument.
 *
 * This typedef can be used as a base class for tensors.
 *
 * @tparam D The tensor type to query.
 */
template <typename D>
using LocationTensorBaseOfT = typename LocationTensorBaseOf<D>::type;

namespace detail {

template <typename T>
constexpr size_t count_of_type() {
    return 0;
}

template <typename T, typename First, typename... Args>
constexpr size_t count_of_type(/*Args... args*/) {
    if constexpr (std::is_same_v<T, First>) {
        return 1 + count_of_type<T, Args...>();
    } else {
        return count_of_type<T, Args...>();
    }
}
} // namespace detail

template <typename T, typename... Args>
concept NoneOfType = !(std::is_same_v<T, std::remove_cvref_t<Args>> || ... || false);

template <typename T, typename... Args>
concept AtLeastOneOfType = (std::is_same_v<T, std::remove_cvref_t<Args>> || ... || false);

template <typename T, size_t Num, typename... Args>
concept NumOfType = detail::count_of_type<T, std::remove_cvref_t<Args>...>() == Num;

template <typename T, typename... Args>
concept AllOfType = (std::is_same_v<T, std::remove_cvref_t<Args>> && ... && true);

#ifdef EINSUMS_COMPUTE_CODE
template <typename T>
using DevDatatype = typename tensor_base::DeviceTypedTensor<T>::dev_datatype;
#endif

} // namespace einsums
