//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

namespace einsums {

/// Helper class that can be passed to a field to trigger the default value of the type.
struct Default {};

inline static const auto default_value = Default{};

}
