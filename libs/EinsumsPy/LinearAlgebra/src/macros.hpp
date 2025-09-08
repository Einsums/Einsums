//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @def EINSUMS_PY_LINALG_CALL
 *
 * Call a piece of code for each of @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 * A condition can be added to make sure the code only runs for one of these. This ends in an empty if statment that checks if
 * any of the expressions ran for the type. This means that error checking code can be placed in an else statement right after this macro.
 * The type being handled is given as a template parameter called @c Float .
 *
 * @param __condition__ The condition to check to make sure that only the necessary code runs.
 * @param __call__ The piece of code to run.
 */
#define EINSUMS_PY_LINALG_CALL(__condition__, __call__)                                                                                    \
    auto __lambda__ = [&]<typename Float>(Float __val__, bool __skip_arg__) {                                                              \
        bool __cond__ = (__condition__);                                                                                                   \
        if (__cond__ && !__skip_arg__) {                                                                                                   \
            (__call__);                                                                                                                    \
        }                                                                                                                                  \
        return __cond__ || __skip_arg__;                                                                                                   \
    };                                                                                                                                     \
    bool __skip__ = __lambda__(float{0.0}, false);                                                                                         \
    __skip__      = __lambda__(double{0.0}, __skip__);                                                                                     \
    __skip__      = __lambda__(std::complex<float>{0.0}, __skip__);                                                                        \
    __skip__      = __lambda__(std::complex<double>{0.0}, __skip__);                                                                       \
    if (__skip__) {                                                                                                                        \
        ; /* Do nothing, but an else can come after this. */                                                                               \
    }
