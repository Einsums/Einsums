//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/StringUtil/MemoryString.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

namespace einsums {
namespace gpu {

extern EINSUMS_EXPORT __constant__ float float_constants[6];
extern EINSUMS_EXPORT  __constant__ double double_constants[6];

namespace detail {

/// @todo This class can be freely changed. It is provided as a starting point for your convenience. If not needed, it may be removed.

class EINSUMS_EXPORT Einsums_GPUMemory_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_GPUMemory_vars)

  public:
    // Put module-global variables here.

    /**
     * @brief Observer to watch for buffer size updates.
     */
    static void update_max_size(config_mapping_type<std::string> const &options);

    /**
     * @brief Resets the current number of allocated bytes to zero.
     */
    void reset_curr_size();

    /**
     * @brief Checks to see if the number of bytes can be allocated.
     *
     * If the number of bytes can not be allocated, returns false. If the number of bytes can be allocated,
     * returns true and updates the current number of allocated bytes.
     */
    bool try_allocate(size_t bytes);

    void deallocate(size_t bytes);

    size_t get_max_size() const;

    float *get_const(float val) {
      if(val == 0.0f) {
        return float_constants;
      } else if(val == 1.0f) {
        return float_constants + 1;
      } else if(val == -1.0f) {
        return float_constants + 3;
      } else {
        return nullptr;
      }
    }

    double *get_const(double val) {
      if(val == 0.0) {
        return double_constants;
      } else if(val == 1.0) {
        return double_constants + 1;
      } else if(val == -1.0) {
        return double_constants + 3;
      } else {
        return nullptr;
      }
    }

    hipFloatComplex *get_const(std::complex<float> val) {
      if(val == 0.0f) {
        return (hipFloatComplex *) (float_constants + 4);
      } else if(val == 1.0f) {
        return (hipFloatComplex *) (float_constants + 1);
      } else if(val == -1.0f) {
        return (hipFloatComplex *) (float_constants + 3);
      } else if(val == std::complex<float>{0.0f, 1.0f}) {
        return (hipFloatComplex *) (float_constants);
      } else if(val == std::complex<float>{0.0f, -1.0f}) {
        return (hipFloatComplex *) (float_constants + 2);
      } else {
        return nullptr;
      }
    }

    hipDoubleComplex *get_const(std::complex<double> val) {
      if(val == 0.0) {
        return (hipDoubleComplex *) (double_constants + 4);
      } else if(val == 1.0) {
        return (hipDoubleComplex *) (double_constants + 1);
      } else if(val == -1.0) {
        return (hipDoubleComplex *) (double_constants + 3);
      } else if(val == std::complex<double>{0.0, 1.0}) {
        return (hipDoubleComplex *) (double_constants);
      } else if(val == std::complex<double>{0.0, -1.0}) {
        return (hipDoubleComplex *) (double_constants + 2);
      } else {
        return nullptr;
      }
    }

  private:
    explicit Einsums_GPUMemory_vars() = default;

    size_t max_size_;
    size_t curr_size_;
};
} // namespace detail
} // namespace gpu
} // namespace einsums