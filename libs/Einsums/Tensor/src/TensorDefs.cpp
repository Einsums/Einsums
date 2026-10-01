//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <H5Lpublic.h>

namespace einsums {

TENSOR_DEFINE_RANK(BlockTensor, 2)
TENSOR_DEFINE_RANK(BlockTensor, 3)
TENSOR_DEFINE_RANK(BlockTensor, 4)

TENSOR_DEFINE(DiskTensor)
TENSOR_DEFINE(DiskView)

TENSOR_DEFINE_RANK(Tensor, 0)
TENSOR_DEFINE(Tensor)
TENSOR_DEFINE(TensorView)

TENSOR_DEFINE(TiledTensor)
TENSOR_DEFINE(TiledTensorView)

#ifndef EINSUMS_WINDOWS
template class RuntimeTensor<float>;
template class RuntimeTensor<double>;
template class RuntimeTensor<std::complex<float>>;
template class RuntimeTensor<std::complex<double>>;

template class RuntimeTensorView<float>;
template class RuntimeTensorView<double>;
template class RuntimeTensorView<std::complex<float>>;
template class RuntimeTensorView<std::complex<double>>;
#endif

namespace detail {
bool verify_exists(hid_t loc_id, std::string const &path, hid_t lapl_id) {
    if (path.length() <= 1) {
        return true;
    }

    std::string temp_path;

    temp_path.reserve(path.length());

    for (auto ch : path) {
        if (ch == '/' && temp_path.length() > 0) {
            auto res = H5Lexists(loc_id, temp_path.c_str(), lapl_id);

            if (res <= 0) {
                return false;
            }
        }
        temp_path.push_back(ch);
    }

    return H5Lexists(loc_id, temp_path.c_str(), lapl_id) > 0;
}
} // namespace detail

} // namespace einsums