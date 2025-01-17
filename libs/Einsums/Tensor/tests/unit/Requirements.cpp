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

#include <Einsums/Testing.hpp>

using namespace einsums;

template <typename T>
void test() {
    static_assert(TensorConcept<BlockTensor<T, 2>>);
    static_assert(!NotTensorConcept<BlockTensor<T, 2>>);
    static_assert(TypedTensorConcept<BlockTensor<T, 2>, T>);
    static_assert(!TypedTensorConcept<BlockTensor<T, 2>, float>);
    static_assert(RankTensorConcept<BlockTensor<T, 2>>);
    static_assert(RankTensorConcept<BlockTensor<T, 2>, 2>);
    static_assert(!RankTensorConcept<BlockTensor<T, 2>, 3>);
    static_assert(BasicLockableConcept<BlockTensor<T, 2>>);
    static_assert(LockableConcept<BlockTensor<T, 2>>);
    static_assert(TRTensorConcept<BlockTensor<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<BlockTensor<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<BlockTensor<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<BlockTensor<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<BlockTensor<T, 2>>);
    static_assert(!TensorViewConcept<BlockTensor<T, 2>>);
    static_assert(!BasicTensorConcept<BlockTensor<T, 2>>);
    static_assert(CollectedTensorConcept<BlockTensor<T, 2>>);
    static_assert(!TiledTensorConcept<BlockTensor<T, 2>>);
    static_assert(!TiledTensorConcept<BlockTensor<T, 2>, Tensor<T, 2>>);
    static_assert(BlockTensorConcept<BlockTensor<T, 2>>);
    static_assert(BlockTensorConcept<BlockTensor<T, 2>, Tensor<T, 2>>);
    static_assert(FunctionTensorConcept<BlockTensor<double, 2>>);
    static_assert(AlgebraTensorConcept<BlockTensor<T, 2>>);

    static_assert(TensorConcept<DiskTensor<T, 2>>);
    static_assert(!NotTensorConcept<DiskTensor<T, 2>>);
    static_assert(TypedTensorConcept<DiskTensor<T, 2>, T>);
    static_assert(!TypedTensorConcept<DiskTensor<T, 2>, float>);
    static_assert(RankTensorConcept<DiskTensor<T, 2>>);
    static_assert(RankTensorConcept<DiskTensor<T, 2>, 2>);
    static_assert(!RankTensorConcept<DiskTensor<T, 2>, 3>);
    static_assert(BasicLockableConcept<DiskTensor<T, 2>>);
    static_assert(LockableConcept<DiskTensor<T, 2>>);
    static_assert(TRTensorConcept<DiskTensor<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<DiskTensor<T, 2>, 2, T>);
    static_assert(!CoreTensorConcept<DiskTensor<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<DiskTensor<T, 2>>);
#endif
    static_assert(DiskTensorConcept<DiskTensor<T, 2>>);
    static_assert(!TensorViewConcept<DiskTensor<T, 2>>);
    static_assert(!BasicTensorConcept<DiskTensor<T, 2>>);
    static_assert(!CollectedTensorConcept<DiskTensor<T, 2>>);
    static_assert(!TiledTensorConcept<DiskTensor<T, 2>>);
    static_assert(!TiledTensorConcept<DiskTensor<T, 2>, Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<DiskTensor<T, 2>>);
    static_assert(!BlockTensorConcept<DiskTensor<T, 2>, Tensor<T, 2>>);
    static_assert(!FunctionTensorConcept<DiskTensor<T, 2>>);
    static_assert(!AlgebraTensorConcept<DiskTensor<T, 2>>);

    static_assert(TensorConcept<DiskView<T, 2, 2>>);
    static_assert(!NotTensorConcept<DiskView<T, 2, 2>>);
    static_assert(TypedTensorConcept<DiskView<T, 2, 2>, T>);
    static_assert(!TypedTensorConcept<DiskView<T, 2, 2>, float>);
    static_assert(RankTensorConcept<DiskView<T, 2, 2>>);
    static_assert(RankTensorConcept<DiskView<T, 2, 2>, 2>);
    static_assert(!RankTensorConcept<DiskView<T, 2, 2>, 3>);
    static_assert(BasicLockableConcept<DiskView<T, 2, 2>>);
    static_assert(LockableConcept<DiskView<T, 2, 2>>);
    static_assert(TRTensorConcept<DiskView<T, 2, 2>, 2, T>);
    static_assert(TRLTensorConcept<DiskView<T, 2, 2>, 2, T>);
    static_assert(!CoreTensorConcept<DiskView<T, 2, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<DiskView<T, 2, 2>>);
#endif
    static_assert(DiskTensorConcept<DiskView<T, 2, 2>>);
    static_assert(TensorViewConcept<DiskView<T, 2, 2>>);
    static_assert(!BasicTensorConcept<DiskView<T, 2, 2>>);
    static_assert(!CollectedTensorConcept<DiskView<T, 2, 2>>);
    static_assert(!TiledTensorConcept<DiskView<T, 2, 2>>);
    static_assert(!TiledTensorConcept<DiskView<T, 2, 2>, Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<DiskView<T, 2, 2>>);
    static_assert(!BlockTensorConcept<DiskView<T, 2, 2>, Tensor<T, 2>>);
    static_assert(FunctionTensorConcept<DiskView<T, 2, 2>>);
    static_assert(!AlgebraTensorConcept<DiskView<T, 2, 2>>);

    static_assert(TensorConcept<KroneckerDelta<T>>);
    static_assert(!NotTensorConcept<KroneckerDelta<T>>);
    static_assert(TypedTensorConcept<KroneckerDelta<T>, T>);
    static_assert(!TypedTensorConcept<KroneckerDelta<T>, float>);
    static_assert(RankTensorConcept<KroneckerDelta<T>>);
    static_assert(RankTensorConcept<KroneckerDelta<T>, 2>);
    static_assert(!RankTensorConcept<KroneckerDelta<T>, 3>);
    static_assert(!BasicLockableConcept<KroneckerDelta<T>>);
    static_assert(!LockableConcept<KroneckerDelta<T>>);
    static_assert(TRTensorConcept<KroneckerDelta<T>, 2, T>);
    static_assert(!TRLTensorConcept<KroneckerDelta<T>, 2, T>);
    static_assert(!CoreTensorConcept<KroneckerDelta<T>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<KroneckerDelta<T>>);
#endif
    static_assert(!DiskTensorConcept<KroneckerDelta<T>>);
    static_assert(!TensorViewConcept<KroneckerDelta<T>>);
    static_assert(!BasicTensorConcept<KroneckerDelta<T>>);
    static_assert(!CollectedTensorConcept<KroneckerDelta<T>>);
    static_assert(!TiledTensorConcept<KroneckerDelta<T>>);
    static_assert(!TiledTensorConcept<KroneckerDelta<T>, Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<KroneckerDelta<T>>);
    static_assert(!BlockTensorConcept<KroneckerDelta<T>, Tensor<T, 2>>);
    static_assert(FunctionTensorConcept<KroneckerDelta<T>>);
    static_assert(!AlgebraTensorConcept<KroneckerDelta<T>>);

    static_assert(TensorConcept<RuntimeTensor<T>>);
    static_assert(!NotTensorConcept<RuntimeTensor<T>>);
    static_assert(TypedTensorConcept<RuntimeTensor<T>, T>);
    static_assert(!TypedTensorConcept<RuntimeTensor<T>, float>);
    static_assert(!RankTensorConcept<RuntimeTensor<T>>);
    static_assert(!RankTensorConcept<RuntimeTensor<T>, 2>);
    static_assert(!RankTensorConcept<RuntimeTensor<T>, 3>);
    static_assert(BasicLockableConcept<RuntimeTensor<T>>);
    static_assert(LockableConcept<RuntimeTensor<T>>);
    static_assert(!TRTensorConcept<RuntimeTensor<T>, 2, T>);
    static_assert(!TRLTensorConcept<RuntimeTensor<T>, 2, T>);
    static_assert(CoreTensorConcept<RuntimeTensor<T>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<RuntimeTensor<T>>);
#endif
    static_assert(!DiskTensorConcept<RuntimeTensor<T>>);
    static_assert(!TensorViewConcept<RuntimeTensor<T>>);
    static_assert(BasicTensorConcept<RuntimeTensor<T>>);
    static_assert(!CollectedTensorConcept<RuntimeTensor<T>>);
    static_assert(!TiledTensorConcept<RuntimeTensor<T>>);
    static_assert(!TiledTensorConcept<RuntimeTensor<T>, Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<RuntimeTensor<T>>);
    static_assert(!BlockTensorConcept<RuntimeTensor<T>, Tensor<T, 2>>);
    static_assert(!FunctionTensorConcept<RuntimeTensor<T>>);
    static_assert(!AlgebraTensorConcept<RuntimeTensor<T>>);

    static_assert(TensorConcept<RuntimeTensorView<T>>);
    static_assert(!NotTensorConcept<RuntimeTensorView<T>>);
    static_assert(TypedTensorConcept<RuntimeTensorView<T>, T>);
    static_assert(!TypedTensorConcept<RuntimeTensorView<T>, float>);
    static_assert(!RankTensorConcept<RuntimeTensorView<T>>);
    static_assert(!RankTensorConcept<RuntimeTensorView<T>, 2>);
    static_assert(!RankTensorConcept<RuntimeTensorView<T>, 3>);
    static_assert(BasicLockableConcept<RuntimeTensorView<T>>);
    static_assert(LockableConcept<RuntimeTensorView<T>>);
    static_assert(!TRTensorConcept<RuntimeTensorView<T>, 2, T>);
    static_assert(!TRLTensorConcept<RuntimeTensorView<T>, 2, T>);
    static_assert(CoreTensorConcept<RuntimeTensorView<T>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<RuntimeTensorView<T>>);
#endif
    static_assert(!DiskTensorConcept<RuntimeTensorView<T>>);
    static_assert(!TensorViewConcept<RuntimeTensorView<T>>);
    static_assert(BasicTensorConcept<RuntimeTensorView<T>>);
    static_assert(!CollectedTensorConcept<RuntimeTensorView<T>>);
    static_assert(!TiledTensorConcept<RuntimeTensorView<T>>);
    static_assert(!TiledTensorConcept<RuntimeTensorView<T>, Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<RuntimeTensorView<T>>);
    static_assert(!BlockTensorConcept<RuntimeTensorView<T>, Tensor<T, 2>>);
    static_assert(!FunctionTensorConcept<RuntimeTensorView<T>>);
    static_assert(!AlgebraTensorConcept<RuntimeTensorView<T>>);

    static_assert(TensorConcept<Tensor<T, 2>>);
    static_assert(!NotTensorConcept<Tensor<T, 2>>);
    static_assert(TypedTensorConcept<Tensor<T, 2>, T>);
    static_assert(!TypedTensorConcept<Tensor<T, 2>, float>);
    static_assert(RankTensorConcept<Tensor<T, 2>>);
    static_assert(RankTensorConcept<Tensor<T, 2>, 2>);
    static_assert(!RankTensorConcept<Tensor<T, 2>, 3>);
    static_assert(BasicLockableConcept<Tensor<T, 2>>);
    static_assert(LockableConcept<Tensor<T, 2>>);
    static_assert(TRTensorConcept<Tensor<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<Tensor<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<Tensor<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<Tensor<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<Tensor<T, 2>>);
    static_assert(!TensorViewConcept<Tensor<T, 2>>);
    static_assert(BasicTensorConcept<Tensor<T, 2>>);
    static_assert(!CollectedTensorConcept<Tensor<T, 2>>);
    static_assert(!TiledTensorConcept<Tensor<T, 2>>);
    static_assert(!TiledTensorConcept<Tensor<T, 2>, Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<Tensor<T, 2>, Tensor<T, 2>>);
    static_assert(FunctionTensorConcept<Tensor<T, 2>>);
    static_assert(AlgebraTensorConcept<Tensor<T, 2>>);

    static_assert(TensorConcept<TensorView<T, 2>>);
    static_assert(!NotTensorConcept<TensorView<T, 2>>);
    static_assert(TypedTensorConcept<TensorView<T, 2>, T>);
    static_assert(!TypedTensorConcept<TensorView<T, 2>, float>);
    static_assert(RankTensorConcept<TensorView<T, 2>>);
    static_assert(RankTensorConcept<TensorView<T, 2>, 2>);
    static_assert(!RankTensorConcept<TensorView<T, 2>, 3>);
    static_assert(BasicLockableConcept<TensorView<T, 2>>);
    static_assert(LockableConcept<TensorView<T, 2>>);
    static_assert(TRTensorConcept<TensorView<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<TensorView<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<TensorView<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<TensorView<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<TensorView<T, 2>>);
    static_assert(TensorViewConcept<TensorView<T, 2>>);
    static_assert(BasicTensorConcept<TensorView<T, 2>>);
    static_assert(!CollectedTensorConcept<TensorView<T, 2>>);
    static_assert(!TiledTensorConcept<TensorView<T, 2>>);
    static_assert(!TiledTensorConcept<TensorView<T, 2>, TensorView<T, 2>>);
    static_assert(!BlockTensorConcept<TensorView<T, 2>>);
    static_assert(!BlockTensorConcept<TensorView<T, 2>, TensorView<T, 2>>);
    static_assert(FunctionTensorConcept<TensorView<T, 2>>);
    static_assert(AlgebraTensorConcept<TensorView<T, 2>>);

    static_assert(TensorConcept<TiledTensor<T, 2>>);
    static_assert(!NotTensorConcept<TiledTensor<T, 2>>);
    static_assert(TypedTensorConcept<TiledTensor<T, 2>, T>);
    static_assert(!TypedTensorConcept<TiledTensor<T, 2>, float>);
    static_assert(RankTensorConcept<TiledTensor<T, 2>>);
    static_assert(RankTensorConcept<TiledTensor<T, 2>, 2>);
    static_assert(!RankTensorConcept<TiledTensor<T, 2>, 3>);
    static_assert(BasicLockableConcept<TiledTensor<T, 2>>);
    static_assert(LockableConcept<TiledTensor<T, 2>>);
    static_assert(TRTensorConcept<TiledTensor<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<TiledTensor<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<TiledTensor<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<TiledTensor<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<TiledTensor<T, 2>>);
    static_assert(!TensorViewConcept<TiledTensor<T, 2>>);
    static_assert(!BasicTensorConcept<TiledTensor<T, 2>>);
    static_assert(CollectedTensorConcept<TiledTensor<T, 2>>);
    static_assert(TiledTensorConcept<TiledTensor<T, 2>>);
    static_assert(TiledTensorConcept<TiledTensor<T, 2>, Tensor<T, 2>>);
    static_assert(!BlockTensorConcept<TiledTensor<T, 2>>);
    static_assert(!BlockTensorConcept<TiledTensor<T, 2>, Tensor<T, 2>>);
    static_assert(FunctionTensorConcept<TiledTensor<T, 2>>);
    static_assert(AlgebraTensorConcept<TiledTensor<T, 2>>);

    static_assert(TensorConcept<TiledTensorView<T, 2>>);
    static_assert(!NotTensorConcept<TiledTensorView<T, 2>>);
    static_assert(TypedTensorConcept<TiledTensorView<T, 2>, T>);
    static_assert(!TypedTensorConcept<TiledTensorView<T, 2>, float>);
    static_assert(RankTensorConcept<TiledTensorView<T, 2>>);
    static_assert(RankTensorConcept<TiledTensorView<T, 2>, 2>);
    static_assert(!RankTensorConcept<TiledTensorView<T, 2>, 3>);
    static_assert(BasicLockableConcept<TiledTensorView<T, 2>>);
    static_assert(LockableConcept<TiledTensorView<T, 2>>);
    static_assert(TRTensorConcept<TiledTensorView<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<TiledTensorView<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<TiledTensorView<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<TiledTensorView<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<TiledTensorView<T, 2>>);
    static_assert(TensorViewConcept<TiledTensorView<T, 2>>);
    static_assert(!BasicTensorConcept<TiledTensorView<T, 2>>);
    static_assert(CollectedTensorConcept<TiledTensorView<T, 2>>);
    static_assert(TiledTensorConcept<TiledTensorView<T, 2>>);
    static_assert(TiledTensorConcept<TiledTensorView<T, 2>, TensorView<T, 2>>);
    static_assert(!BlockTensorConcept<TiledTensorView<T, 2>>);
    static_assert(!BlockTensorConcept<TiledTensorView<T, 2>, Tensor<T, 2>>);
    static_assert(FunctionTensorConcept<TiledTensorView<T, 2>>);
    static_assert(AlgebraTensorConcept<TiledTensorView<T, 2>>);

#ifdef EINSUMS_COMPUTE_CODE
static_assert(TensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(!NotTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(TypedTensorConcept<BlockDeviceTensor<T, 2>, T>);
    static_assert(!TypedTensorConcept<BlockDeviceTensor<T, 2>, float>);
    static_assert(RankTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(RankTensorConcept<BlockDeviceTensor<T, 2>, 2>);
    static_assert(!RankTensorConcept<BlockDeviceTensor<T, 2>, 3>);
    static_assert(BasicLockableConcept<BlockDeviceTensor<T, 2>>);
    static_assert(LockableConcept<BlockDeviceTensor<T, 2>>);
    static_assert(TRTensorConcept<BlockDeviceTensor<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<BlockDeviceTensor<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(!DeviceTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(!DiskTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(!TensorViewConcept<BlockDeviceTensor<T, 2>>);
    static_assert(!BasicTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(CollectedTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(!TiledTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(!TiledTensorConcept<BlockDeviceTensor<T, 2>, DeviceTensor<T, 2>>);
    static_assert(BlockTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(BlockTensorConcept<BlockDeviceTensor<T, 2>, DeviceTensor<T, 2>>);
    static_assert(FunctionTensorConcept<BlockDeviceTensor<T, 2>>);
    static_assert(AlgebraTensorConcept<BlockDeviceTensor<T, 2>>);

    static_assert(TensorConcept<DeviceTensor<T, 2>>);
    static_assert(!NotTensorConcept<DeviceTensor<T, 2>>);
    static_assert(TypedTensorConcept<DeviceTensor<T, 2>, T>);
    static_assert(!TypedTensorConcept<DeviceTensor<T, 2>, float>);
    static_assert(RankTensorConcept<DeviceTensor<T, 2>>);
    static_assert(RankTensorConcept<DeviceTensor<T, 2>, 2>);
    static_assert(!RankTensorConcept<DeviceTensor<T, 2>, 3>);
    static_assert(BasicLockableConcept<DeviceTensor<T, 2>>);
    static_assert(LockableConcept<DeviceTensor<T, 2>>);
    static_assert(TRTensorConcept<DeviceTensor<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<DeviceTensor<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<DeviceTensor<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<DeviceTensor<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<DeviceTensor<T, 2>>);
    static_assert(!TensorViewConcept<DeviceTensor<T, 2>>);
    static_assert(BasicTensorConcept<DeviceTensor<T, 2>>);
    static_assert(!CollectedTensorConcept<DeviceTensor<T, 2>>);
    static_assert(!TiledTensorConcept<DeviceTensor<T, 2>>);
    static_assert(!TiledTensorConcept<DeviceTensor<T, 2>, DeviceTensor<T, 2>>);
    static_assert(!BlockTensorConcept<DeviceTensor<T, 2>>);
    static_assert(!BlockTensorConcept<DeviceTensor<T, 2>, DeviceTensor<T, 2>>);
    static_assert(FunctionTensorConcept<DeviceTensor<T, 2>>);
    static_assert(AlgebraTensorConcept<DeviceTensor<T, 2>>);

    static_assert(TensorConcept<DeviceTensorView<T, 2>>);
    static_assert(!NotTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(TypedTensorConcept<DeviceTensorView<T, 2>, T>);
    static_assert(!TypedTensorConcept<DeviceTensorView<T, 2>, float>);
    static_assert(RankTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(RankTensorConcept<DeviceTensorView<T, 2>, 2>);
    static_assert(!RankTensorConcept<DeviceTensorView<T, 2>, 3>);
    static_assert(BasicLockableConcept<DeviceTensorView<T, 2>>);
    static_assert(LockableConcept<DeviceTensorView<T, 2>>);
    static_assert(TRTensorConcept<DeviceTensorView<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<DeviceTensorView<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<DeviceTensorView<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<DeviceTensorView<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(TensorViewConcept<DeviceTensorView<T, 2>>);
    static_assert(BasicTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(!CollectedTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(!TiledTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(!TiledTensorConcept<DeviceTensorView<T, 2>, DeviceTensorView<T, 2>>);
    static_assert(!BlockTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(!BlockTensorConcept<DeviceTensorView<T, 2>, DeviceTensorView<T, 2>>);
    static_assert(FunctionTensorConcept<DeviceTensorView<T, 2>>);
    static_assert(AlgebraTensorConcept<DeviceTensorView<T, 2>>);

    static_assert(TensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(!NotTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(TypedTensorConcept<TiledDeviceTensor<T, 2>, T>);
    static_assert(!TypedTensorConcept<TiledDeviceTensor<T, 2>, float>);
    static_assert(RankTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(RankTensorConcept<TiledDeviceTensor<T, 2>, 2>);
    static_assert(!RankTensorConcept<TiledDeviceTensor<T, 2>, 3>);
    static_assert(BasicLockableConcept<TiledDeviceTensor<T, 2>>);
    static_assert(LockableConcept<TiledDeviceTensor<T, 2>>);
    static_assert(TRTensorConcept<TiledDeviceTensor<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<TiledDeviceTensor<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<TiledDeviceTensor<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<TiledDeviceTensor<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(!TensorViewConcept<TiledDeviceTensor<T, 2>>);
    static_assert(!BasicTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(CollectedTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(TiledTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(TiledTensorConcept<TiledDeviceTensor<T, 2>, DeviceTensor<T, 2>>);
    static_assert(!BlockTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(!BlockTensorConcept<TiledDeviceTensor<T, 2>, DeviceTensor<T, 2>>);
    static_assert(FunctionTensorConcept<TiledDeviceTensor<T, 2>>);
    static_assert(AlgebraTensorConcept<TiledDeviceTensor<T, 2>>);

    static_assert(TensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(!NotTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(TypedTensorConcept<TiledDeviceTensorView<T, 2>, T>);
    static_assert(!TypedTensorConcept<TiledDeviceTensorView<T, 2>, float>);
    static_assert(RankTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(RankTensorConcept<TiledDeviceTensorView<T, 2>, 2>);
    static_assert(!RankTensorConcept<TiledDeviceTensorView<T, 2>, 3>);
    static_assert(BasicLockableConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(LockableConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(TRTensorConcept<TiledDeviceTensorView<T, 2>, 2, T>);
    static_assert(TRLTensorConcept<TiledDeviceTensorView<T, 2>, 2, T>);
    static_assert(CoreTensorConcept<TiledDeviceTensorView<T, 2>>);
#ifdef EINSUMS_COMPUTE_CODE
    static_assert(!DeviceTensorConcept<TiledDeviceTensorView<T, 2>>);
#endif
    static_assert(!DiskTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(TensorViewConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(!BasicTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(CollectedTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(TiledTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(TiledTensorConcept<TiledDeviceTensorView<T, 2>, DeviceTensorView<T, 2>>);
    static_assert(!BlockTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(!BlockTensorConcept<TiledDeviceTensorView<T, 2>, DeviceTensor<T, 2>>);
    static_assert(FunctionTensorConcept<TiledDeviceTensorView<T, 2>>);
    static_assert(AlgebraTensorConcept<TiledDeviceTensorView<T, 2>>);
#endif

}

TEST_CASE("Tensor requirements", "[tensor]") {
    test<double>();
}