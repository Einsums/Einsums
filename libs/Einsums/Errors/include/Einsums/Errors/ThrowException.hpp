//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Preprocessor/Expand.hpp>
#include <Einsums/Preprocessor/NArgs.hpp>

#include <exception>
#include <source_location>
#include <string>
#include <system_error>

#define EINSUMS_THROW_STRINGIFY(arg) #arg

#define EINSUMS_THROW_STD_EXCEPTION(except)                                                                                                \
    throw except(einsums::detail::make_error_message(EINSUMS_THROW_STRINGIFY(except), "", std::source_location::current())) /**/

#define EINSUMS_THROW_EXCEPTION(except, ...)                                                                                               \
    throw except(einsums::detail::make_error_message(EINSUMS_THROW_STRINGIFY(except), fmt::format(__VA_ARGS__),                            \
                                                     std::source_location::current())) /**/