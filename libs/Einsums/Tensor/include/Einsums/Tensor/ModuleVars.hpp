//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <H5Ipublic.h>
#include <Einsums/TypeSupport/Singleton.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/Config.hpp>

namespace einsums {
namespace detail {

/// @todo This class can be freely changed. It is provided as a starting point for your convenience. If not needed, it may be removed.

class EINSUMS_EXPORT Einsums_Tensor_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_Tensor_vars)

public:
    // Put module-global variables here.
    hid_t hdf5_file;
    hid_t link_property_list;

    hid_t double_complex_type;
    hid_t float_complex_type;

private:
    explicit Einsums_Tensor_vars() = default;

};

}
}