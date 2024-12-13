#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>


namespace einsums {

TENSOR_DEFINE_RANK(BlockDeviceTensor, 2)
TENSOR_DEFINE_RANK(BlockDeviceTensor, 3)
TENSOR_DEFINE_RANK(BlockDeviceTensor, 4)

TENSOR_DEFINE_RANK(DeviceTensor, 0)
TENSOR_DEFINE(DeviceTensor)
TENSOR_DEFINE(DeviceTensorView)

}