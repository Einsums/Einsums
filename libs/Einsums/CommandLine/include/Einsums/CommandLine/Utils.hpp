//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums {
namespace cl {

struct Range {
    long long min_v = (std::numeric_limits<long long>::min)();
    long long max_v = (std::numeric_limits<long long>::max)();

    constexpr Range(long long begin, long long end) : min_v { begin }, max_v { end } {
    }
};

enum ParseResult {
    SUCCESS = 0,
    HELP,
    VERSION,
    MISSING_REQUIRED,
    INVALID_ARGUMENT,
    INCOMPATIBLE_ARGUMENT,
    UNKNOWN_ARGUMENT,
    CONFIG_ERROR
};

}
}
