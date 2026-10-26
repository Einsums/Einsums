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

#include <H5Lpublic.h>
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace einsums {

static bool verify_path(std::string const &path) {
    if (path.size() == 0) {
        return true;
    }

    if (path[0] != '/' && path[0] != '.') {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "The format of the disk tensor name \"{}\" was invalid! It must be formatted as a path.", path);
    }

    return true;
}

namespace detail {

bool verify_exists(hid_t loc_id, std::string const &path, hid_t lapl_id) {
    if (!verify_path(path)) {
        return false;
    }
    if (path.length() == 0) {
        return false;
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