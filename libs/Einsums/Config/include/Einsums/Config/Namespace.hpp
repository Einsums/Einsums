//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#define EINSUMS_BEGIN_NAMESPACE_HPP(x)                                                                                                     \
    namespace x {                                                                                                                          \
    namespace detail {}                                                                                                                    \
    }

#define EINSUMS_END_NAMESPACE_HPP(x) }

#define EINSUMS_BEGIN_NAMESPACE_CPP(x)                                                                                                     \
    namespace x {                                                                                                                          \
    namespace detail {                                                                                                                     \
    namespace {}                                                                                                                           \
    }                                                                                                                                      \
    }

#define EINSUMS_END_NAMESPACE_CPP(x) }
