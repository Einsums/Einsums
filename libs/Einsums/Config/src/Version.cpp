//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Version.hpp>
#include <Einsums/Preprocessor/Stringify.hpp>

namespace einsums {

char const EINSUMS_CHECK_VERSION[] = EINSUMS_PP_STRINGIFY(EINSUMS_CHECK_VERSION);

}