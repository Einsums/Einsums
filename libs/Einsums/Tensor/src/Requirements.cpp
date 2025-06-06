#include <Einsums/Config.hpp>

#include <Einsums/Tensor/ArithmeticTensor.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/FunctionTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/Tensor/DeviceTensor.hpp>
#endif

using namespace einsums;

/**
 * @brief Checks to make sure that all tensors satisfy the concepts they are expected to and none of the ones they are not expected to.
 *
 * Contains a whole bunch of static asserts. Each assertion checks whether a tensor satisfies a concept or does not satisfy a concept.
 * This is to ensure that changes to concepts or tensors will still keep compile-time deduction as it should be expected to perform.
 * For instance, we wouldn't want a change to Tensor to make it so it could not be used with an einsum call.
 */
template <typename T, typename BadT, size_t Rank, size_t BadRank>
static void check_requirements() {
    if constexpr (std::is_same_v<T, BadT> || Rank == BadRank) {
        return;
    } else {
        if constexpr (Rank >= 2) {
            static_assert(TensorConcept<BlockTensor<T, Rank>>);
            static_assert(!NotTensorConcept<BlockTensor<T, Rank>>);
            static_assert(TypedTensorConcept<BlockTensor<T, Rank>, T>);
            static_assert(!TypedTensorConcept<BlockTensor<T, Rank>, BadT>);
            static_assert(RankTensorConcept<BlockTensor<T, Rank>>);
            static_assert(RankTensorConcept<BlockTensor<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<BlockTensor<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<BlockTensor<T, Rank>>);
            static_assert(LockableConcept<BlockTensor<T, Rank>>);
            static_assert(TRTensorConcept<BlockTensor<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<BlockTensor<T, Rank>, Rank, T>);
            static_assert(CoreTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!DeviceTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!DiskTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!TensorViewConcept<BlockTensor<T, Rank>>);
            static_assert(!BasicTensorConcept<BlockTensor<T, Rank>>);
            static_assert(CollectedTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!TiledTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!TiledTensorConcept<BlockTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(BlockTensorConcept<BlockTensor<T, Rank>>);
            static_assert(BlockTensorConcept<BlockTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<BlockTensor<T, Rank>>);
            static_assert(AlgebraTensorConcept<BlockTensor<T, Rank>>);
            static_assert(requires(BlockTensor<T, Rank> tensor) { println(tensor); });
        }

        static_assert(TensorConcept<DiskTensor<T, Rank>>);
        static_assert(!NotTensorConcept<DiskTensor<T, Rank>>);
        static_assert(TypedTensorConcept<DiskTensor<T, Rank>, T>);
        static_assert(!TypedTensorConcept<DiskTensor<T, Rank>, BadT>);
        static_assert(RankTensorConcept<DiskTensor<T, Rank>>);
        static_assert(RankTensorConcept<DiskTensor<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<DiskTensor<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<DiskTensor<T, Rank>>);
        static_assert(LockableConcept<DiskTensor<T, Rank>>);
        static_assert(TRTensorConcept<DiskTensor<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<DiskTensor<T, Rank>, Rank, T>);
        static_assert(!CoreTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!DeviceTensorConcept<DiskTensor<T, Rank>>);
        static_assert(DiskTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!TensorViewConcept<DiskTensor<T, Rank>>);
        static_assert(!BasicTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!CollectedTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskTensor<T, Rank>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskTensor<T, Rank>, Tensor<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(!FunctionTensorConcept<DiskTensor<T, Rank>>);
        }
        static_assert(!AlgebraTensorConcept<DiskTensor<T, Rank>>);
        static_assert(requires(DiskTensor<T, Rank> tensor) { println(tensor); });

        static_assert(TensorConcept<DiskView<T, Rank>>);
        static_assert(!NotTensorConcept<DiskView<T, Rank>>);
        static_assert(TypedTensorConcept<DiskView<T, Rank>, T>);
        static_assert(!TypedTensorConcept<DiskView<T, Rank>, BadT>);
        static_assert(RankTensorConcept<DiskView<T, Rank>>);
        static_assert(RankTensorConcept<DiskView<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<DiskView<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<DiskView<T, Rank>>);
        static_assert(LockableConcept<DiskView<T, Rank>>);
        static_assert(TRTensorConcept<DiskView<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<DiskView<T, Rank>, Rank, T>);
        static_assert(!CoreTensorConcept<DiskView<T, Rank>>);
        static_assert(!DeviceTensorConcept<DiskView<T, Rank>>);
        static_assert(DiskTensorConcept<DiskView<T, Rank>>);
        static_assert(TensorViewConcept<DiskView<T, Rank>>);
        static_assert(!BasicTensorConcept<DiskView<T, Rank>>);
        static_assert(!CollectedTensorConcept<DiskView<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskView<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskView<T, Rank>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskView<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskView<T, Rank>, Tensor<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<DiskView<T, Rank>>);
        }
        static_assert(!AlgebraTensorConcept<DiskView<T, Rank>>);
        static_assert(requires(DiskView<T, Rank> tensor) { println(tensor); });

        if constexpr (Rank == 2) {
            static_assert(TensorConcept<KroneckerDelta<T>>);
            static_assert(!NotTensorConcept<KroneckerDelta<T>>);
            static_assert(TypedTensorConcept<KroneckerDelta<T>, T>);
            static_assert(!TypedTensorConcept<KroneckerDelta<T>, BadT>);
            static_assert(RankTensorConcept<KroneckerDelta<T>>);
            static_assert(RankTensorConcept<KroneckerDelta<T>, Rank>);
            static_assert(!RankTensorConcept<KroneckerDelta<T>, BadRank>);
            static_assert(!BasicLockableConcept<KroneckerDelta<T>>);
            static_assert(!LockableConcept<KroneckerDelta<T>>);
            static_assert(TRTensorConcept<KroneckerDelta<T>, Rank, T>);
            static_assert(!TRLTensorConcept<KroneckerDelta<T>, Rank, T>);
            static_assert(!CoreTensorConcept<KroneckerDelta<T>>);
            static_assert(!DeviceTensorConcept<KroneckerDelta<T>>);
            static_assert(!DiskTensorConcept<KroneckerDelta<T>>);
            static_assert(!TensorViewConcept<KroneckerDelta<T>>);
            static_assert(!BasicTensorConcept<KroneckerDelta<T>>);
            static_assert(!CollectedTensorConcept<KroneckerDelta<T>>);
            static_assert(!TiledTensorConcept<KroneckerDelta<T>>);
            static_assert(!TiledTensorConcept<KroneckerDelta<T>, Tensor<T, Rank>>);
            static_assert(!BlockTensorConcept<KroneckerDelta<T>>);
            static_assert(!BlockTensorConcept<KroneckerDelta<T>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<KroneckerDelta<T>>);
            static_assert(!AlgebraTensorConcept<KroneckerDelta<T>>);
            static_assert(requires(KroneckerDelta<T> tensor) { println(tensor); });
        }

        static_assert(TensorConcept<RuntimeTensor<T>>);
        static_assert(!NotTensorConcept<RuntimeTensor<T>>);
        static_assert(TypedTensorConcept<RuntimeTensor<T>, T>);
        static_assert(!TypedTensorConcept<RuntimeTensor<T>, BadT>);
        static_assert(!RankTensorConcept<RuntimeTensor<T>>);
        static_assert(!RankTensorConcept<RuntimeTensor<T>, Rank>);
        static_assert(!RankTensorConcept<RuntimeTensor<T>, BadRank>);
        static_assert(BasicLockableConcept<RuntimeTensor<T>>);
        static_assert(LockableConcept<RuntimeTensor<T>>);
        static_assert(!TRTensorConcept<RuntimeTensor<T>, Rank, T>);
        static_assert(!TRLTensorConcept<RuntimeTensor<T>, Rank, T>);
        static_assert(CoreTensorConcept<RuntimeTensor<T>>);
        static_assert(!DeviceTensorConcept<RuntimeTensor<T>>);
        static_assert(!DiskTensorConcept<RuntimeTensor<T>>);
        static_assert(!TensorViewConcept<RuntimeTensor<T>>);
        static_assert(BasicTensorConcept<RuntimeTensor<T>>);
        static_assert(!CollectedTensorConcept<RuntimeTensor<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensor<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensor<T>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<RuntimeTensor<T>>);
        static_assert(!BlockTensorConcept<RuntimeTensor<T>, Tensor<T, Rank>>);
        static_assert(!FunctionTensorConcept<RuntimeTensor<T>>);
        static_assert(!AlgebraTensorConcept<RuntimeTensor<T>>);
        static_assert(requires(RuntimeTensor<T> tensor) { println(tensor); });

        static_assert(TensorConcept<RuntimeTensorView<T>>);
        static_assert(!NotTensorConcept<RuntimeTensorView<T>>);
        static_assert(TypedTensorConcept<RuntimeTensorView<T>, T>);
        static_assert(!TypedTensorConcept<RuntimeTensorView<T>, BadT>);
        static_assert(!RankTensorConcept<RuntimeTensorView<T>>);
        static_assert(!RankTensorConcept<RuntimeTensorView<T>, Rank>);
        static_assert(!RankTensorConcept<RuntimeTensorView<T>, BadRank>);
        static_assert(BasicLockableConcept<RuntimeTensorView<T>>);
        static_assert(LockableConcept<RuntimeTensorView<T>>);
        static_assert(!TRTensorConcept<RuntimeTensorView<T>, Rank, T>);
        static_assert(!TRLTensorConcept<RuntimeTensorView<T>, Rank, T>);
        static_assert(CoreTensorConcept<RuntimeTensorView<T>>);
        static_assert(!DeviceTensorConcept<RuntimeTensorView<T>>);
        static_assert(!DiskTensorConcept<RuntimeTensorView<T>>);
        static_assert(!TensorViewConcept<RuntimeTensorView<T>>);
        static_assert(BasicTensorConcept<RuntimeTensorView<T>>);
        static_assert(!CollectedTensorConcept<RuntimeTensorView<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensorView<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensorView<T>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<RuntimeTensorView<T>>);
        static_assert(!BlockTensorConcept<RuntimeTensorView<T>, Tensor<T, Rank>>);
        static_assert(!FunctionTensorConcept<RuntimeTensorView<T>>);
        static_assert(!AlgebraTensorConcept<RuntimeTensorView<T>>);
        static_assert(requires(RuntimeTensorView<T> tensor) { println(tensor); });

        static_assert(TensorConcept<Tensor<T, Rank>>);
        static_assert(!NotTensorConcept<Tensor<T, Rank>>);
        static_assert(TypedTensorConcept<Tensor<T, Rank>, T>);
        static_assert(!TypedTensorConcept<Tensor<T, Rank>, BadT>);
        static_assert(RankTensorConcept<Tensor<T, Rank>>);
        static_assert(RankTensorConcept<Tensor<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<Tensor<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<Tensor<T, Rank>>);
        static_assert(LockableConcept<Tensor<T, Rank>>);
        static_assert(TRTensorConcept<Tensor<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<Tensor<T, Rank>, Rank, T>);
        static_assert(CoreTensorConcept<Tensor<T, Rank>>);
        static_assert(!DeviceTensorConcept<Tensor<T, Rank>>);
        static_assert(!DiskTensorConcept<Tensor<T, Rank>>);
        static_assert(!TensorViewConcept<Tensor<T, Rank>>);
        static_assert(BasicTensorConcept<Tensor<T, Rank>>);
        static_assert(!CollectedTensorConcept<Tensor<T, Rank>>);
        static_assert(!TiledTensorConcept<Tensor<T, Rank>>);
        static_assert(!TiledTensorConcept<Tensor<T, Rank>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<Tensor<T, Rank>, Tensor<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<Tensor<T, Rank>>);
        }
        static_assert(AlgebraTensorConcept<Tensor<T, Rank>>);
        static_assert(requires(Tensor<T, Rank> tensor) { println(tensor); });

        static_assert(TensorConcept<TensorView<T, Rank>>);
        static_assert(!NotTensorConcept<TensorView<T, Rank>>);
        static_assert(TypedTensorConcept<TensorView<T, Rank>, T>);
        static_assert(!TypedTensorConcept<TensorView<T, Rank>, BadT>);
        static_assert(RankTensorConcept<TensorView<T, Rank>>);
        static_assert(RankTensorConcept<TensorView<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<TensorView<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<TensorView<T, Rank>>);
        static_assert(LockableConcept<TensorView<T, Rank>>);
        static_assert(TRTensorConcept<TensorView<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<TensorView<T, Rank>, Rank, T>);
        static_assert(CoreTensorConcept<TensorView<T, Rank>>);
        static_assert(!DeviceTensorConcept<TensorView<T, Rank>>);
        static_assert(!DiskTensorConcept<TensorView<T, Rank>>);
        static_assert(TensorViewConcept<TensorView<T, Rank>>);
        static_assert(BasicTensorConcept<TensorView<T, Rank>>);
        static_assert(!CollectedTensorConcept<TensorView<T, Rank>>);
        static_assert(!TiledTensorConcept<TensorView<T, Rank>>);
        static_assert(!TiledTensorConcept<TensorView<T, Rank>, TensorView<T, Rank>>);
        static_assert(!BlockTensorConcept<TensorView<T, Rank>>);
        static_assert(!BlockTensorConcept<TensorView<T, Rank>, TensorView<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<TensorView<T, Rank>>);
        }
        static_assert(AlgebraTensorConcept<TensorView<T, Rank>>);
        static_assert(requires(TensorView<T, Rank> tensor) { println(tensor); });

        if constexpr (Rank > 0) {
            static_assert(TensorConcept<TiledTensor<T, Rank>>);
            static_assert(!NotTensorConcept<TiledTensor<T, Rank>>);
            static_assert(TypedTensorConcept<TiledTensor<T, Rank>, T>);
            static_assert(!TypedTensorConcept<TiledTensor<T, Rank>, BadT>);
            static_assert(RankTensorConcept<TiledTensor<T, Rank>>);
            static_assert(RankTensorConcept<TiledTensor<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<TiledTensor<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<TiledTensor<T, Rank>>);
            static_assert(LockableConcept<TiledTensor<T, Rank>>);
            static_assert(TRTensorConcept<TiledTensor<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<TiledTensor<T, Rank>, Rank, T>);
            static_assert(CoreTensorConcept<TiledTensor<T, Rank>>);
            static_assert(!DeviceTensorConcept<TiledTensor<T, Rank>>);
            static_assert(!DiskTensorConcept<TiledTensor<T, Rank>>);
            static_assert(!TensorViewConcept<TiledTensor<T, Rank>>);
            static_assert(!BasicTensorConcept<TiledTensor<T, Rank>>);
            static_assert(CollectedTensorConcept<TiledTensor<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensor<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensor<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<TiledTensor<T, Rank>>);
            static_assert(AlgebraTensorConcept<TiledTensor<T, Rank>>);
            static_assert(requires(TiledTensor<T, Rank> tensor) { println(tensor); });

            static_assert(TensorConcept<TiledTensorView<T, Rank>>);
            static_assert(!NotTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TypedTensorConcept<TiledTensorView<T, Rank>, T>);
            static_assert(!TypedTensorConcept<TiledTensorView<T, Rank>, BadT>);
            static_assert(RankTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(RankTensorConcept<TiledTensorView<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<TiledTensorView<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<TiledTensorView<T, Rank>>);
            static_assert(LockableConcept<TiledTensorView<T, Rank>>);
            static_assert(TRTensorConcept<TiledTensorView<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<TiledTensorView<T, Rank>, Rank, T>);
            static_assert(CoreTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(!DeviceTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(!DiskTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TensorViewConcept<TiledTensorView<T, Rank>>);
            static_assert(!BasicTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(CollectedTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensorView<T, Rank>, TensorView<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensorView<T, Rank>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(AlgebraTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(requires(TiledTensorView<T, Rank> tensor) { println(tensor); });
        }

#ifdef EINSUMS_COMPUTE_CODE
        if constexpr (Rank >= 2) {
            static_assert(TensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(!NotTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(TypedTensorConcept<BlockDeviceTensor<T, Rank>, T>);
            static_assert(!TypedTensorConcept<BlockDeviceTensor<T, Rank>, BadT>);
            static_assert(RankTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(RankTensorConcept<BlockDeviceTensor<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<BlockDeviceTensor<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(LockableConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(TRTensorConcept<BlockDeviceTensor<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<BlockDeviceTensor<T, Rank>, Rank, T>);
            static_assert(!CoreTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(DeviceTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(!DiskTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(!TensorViewConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(!BasicTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(CollectedTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(!TiledTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(!TiledTensorConcept<BlockDeviceTensor<T, Rank>, DeviceTensor<T, Rank>>);
            static_assert(BlockTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(BlockTensorConcept<BlockDeviceTensor<T, Rank>, DeviceTensor<T, Rank>>);
            static_assert(FunctionTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(AlgebraTensorConcept<BlockDeviceTensor<T, Rank>>);
            static_assert(requires(BlockDeviceTensor<T, Rank> tensor) { println(tensor); });
        }

        static_assert(TensorConcept<DeviceTensor<T, Rank>>);
        static_assert(!NotTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(TypedTensorConcept<DeviceTensor<T, Rank>, T>);
        static_assert(!TypedTensorConcept<DeviceTensor<T, Rank>, BadT>);
        static_assert(RankTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(RankTensorConcept<DeviceTensor<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<DeviceTensor<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<DeviceTensor<T, Rank>>);
        static_assert(LockableConcept<DeviceTensor<T, Rank>>);
        static_assert(TRTensorConcept<DeviceTensor<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<DeviceTensor<T, Rank>, Rank, T>);
        static_assert(!CoreTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(DeviceTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(!DiskTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(!TensorViewConcept<DeviceTensor<T, Rank>>);
        static_assert(BasicTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(!CollectedTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(!TiledTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(!TiledTensorConcept<DeviceTensor<T, Rank>, DeviceTensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DeviceTensor<T, Rank>, DeviceTensor<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<DeviceTensor<T, Rank>>);
        }
        static_assert(AlgebraTensorConcept<DeviceTensor<T, Rank>>);
        static_assert(requires(DeviceTensor<T, Rank> tensor) { println(tensor); });

        static_assert(TensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(!NotTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(TypedTensorConcept<DeviceTensorView<T, Rank>, T>);
        static_assert(!TypedTensorConcept<DeviceTensorView<T, Rank>, BadT>);
        static_assert(RankTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(RankTensorConcept<DeviceTensorView<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<DeviceTensorView<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<DeviceTensorView<T, Rank>>);
        static_assert(LockableConcept<DeviceTensorView<T, Rank>>);
        static_assert(TRTensorConcept<DeviceTensorView<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<DeviceTensorView<T, Rank>, Rank, T>);
        static_assert(!CoreTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(DeviceTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(!DiskTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(TensorViewConcept<DeviceTensorView<T, Rank>>);
        static_assert(BasicTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(!CollectedTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(!TiledTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(!TiledTensorConcept<DeviceTensorView<T, Rank>, DeviceTensorView<T, Rank>>);
        static_assert(!BlockTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(!BlockTensorConcept<DeviceTensorView<T, Rank>, DeviceTensorView<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<DeviceTensorView<T, Rank>>);
        }
        static_assert(AlgebraTensorConcept<DeviceTensorView<T, Rank>>);
        static_assert(requires(DeviceTensorView<T, Rank> tensor) { println(tensor); });

        if constexpr (Rank > 0) {
            static_assert(TensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(!NotTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(TypedTensorConcept<TiledDeviceTensor<T, Rank>, T>);
            static_assert(!TypedTensorConcept<TiledDeviceTensor<T, Rank>, BadT>);
            static_assert(RankTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(RankTensorConcept<TiledDeviceTensor<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<TiledDeviceTensor<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(LockableConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(TRTensorConcept<TiledDeviceTensor<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<TiledDeviceTensor<T, Rank>, Rank, T>);
            static_assert(!CoreTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(DeviceTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(!DiskTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(!TensorViewConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(!BasicTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(CollectedTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(TiledTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(TiledTensorConcept<TiledDeviceTensor<T, Rank>, DeviceTensor<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledDeviceTensor<T, Rank>, DeviceTensor<T, Rank>>);
            static_assert(FunctionTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(AlgebraTensorConcept<TiledDeviceTensor<T, Rank>>);
            static_assert(requires(TiledDeviceTensor<T, Rank> tensor) { println(tensor); });

            static_assert(TensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(!NotTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(TypedTensorConcept<TiledDeviceTensorView<T, Rank>, T>);
            static_assert(!TypedTensorConcept<TiledDeviceTensorView<T, Rank>, BadT>);
            static_assert(RankTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(RankTensorConcept<TiledDeviceTensorView<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<TiledDeviceTensorView<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(LockableConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(TRTensorConcept<TiledDeviceTensorView<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<TiledDeviceTensorView<T, Rank>, Rank, T>);
            static_assert(!CoreTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(DeviceTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(!DiskTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(TensorViewConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(!BasicTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(CollectedTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(TiledTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(TiledTensorConcept<TiledDeviceTensorView<T, Rank>, DeviceTensorView<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledDeviceTensorView<T, Rank>, DeviceTensor<T, Rank>>);
            static_assert(FunctionTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(AlgebraTensorConcept<TiledDeviceTensorView<T, Rank>>);
            static_assert(requires(TiledDeviceTensorView<T, Rank> tensor) { println(tensor); });
        }
#endif
    }
}

#define CHECK_REQUIREMENTS0(t1, t2, rank1)                                                                                                 \
    check_requirements<t1, t2, rank1, 0>();                                                                                                \
    check_requirements<t1, t2, rank1, 1>();                                                                                                \
    check_requirements<t1, t2, rank1, 2>();                                                                                                \
    check_requirements<t1, t2, rank1, 3>();

#define CHECK_REQUIREMENTS1(t1, t2)                                                                                                        \
    CHECK_REQUIREMENTS0(t1, t2, 0)                                                                                                         \
    CHECK_REQUIREMENTS0(t1, t2, 1)                                                                                                         \
    CHECK_REQUIREMENTS0(t1, t2, 2)

#define CHECK_REQUIREMENTS2(t1)                                                                                                            \
    CHECK_REQUIREMENTS1(t1, float)                                                                                                         \
    CHECK_REQUIREMENTS1(t1, double)                                                                                                        \
    CHECK_REQUIREMENTS1(t1, std::complex<float>)                                                                                           \
    CHECK_REQUIREMENTS1(t1, std::complex<double>)

static void check_requirements_instantiate() {
    CHECK_REQUIREMENTS2(float)
    CHECK_REQUIREMENTS2(double)
    CHECK_REQUIREMENTS2(std::complex<float>)
    CHECK_REQUIREMENTS2(std::complex<double>)
}