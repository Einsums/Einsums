#include <Einsums/Config.hpp>

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace einsums {


// template class RuntimeTensor<float>;
// template class RuntimeTensor<double>;
// template class RuntimeTensor<std::complex<float>>;
// template class RuntimeTensor<std::complex<double>>;

// template class RuntimeTensorView<float>;
// template class RuntimeTensorView<double>;
// template class RuntimeTensorView<std::complex<float>>;
// template class RuntimeTensorView<std::complex<double>>;

TENSOR_DEFINE_RANK(BlockTensor, 2)
TENSOR_DEFINE_RANK(BlockTensor, 3)
TENSOR_DEFINE_RANK(BlockTensor, 4)

TENSOR_DEFINE(DiskTensor)
TENSOR_DEFINE_DISK_VIEW(DiskView)

TENSOR_DEFINE_RANK(Tensor, 0)
TENSOR_DEFINE(Tensor)
TENSOR_DEFINE(TensorView)

TENSOR_DEFINE(TiledTensor)
TENSOR_DEFINE(TiledTensorView)

} // namespace einsums