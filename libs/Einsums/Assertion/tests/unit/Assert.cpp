//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Assert.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("assert") {
    EINSUMS_ASSERT(1 == 0);
}