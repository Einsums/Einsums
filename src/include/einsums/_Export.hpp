//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/*! \def _EXPORT
    Internal macro that defines export visibility based on platform.
*/
#if defined(_WIN32) || defined(__CYGWIN__)
#    define _EXPORT __declspec(dllexport)
#else
#    define _EXPORT __attribute__((visibility("default")))
#endif

/*! \def EINSUMS_EXPORT
    Macro used to decorate classes/functions/structs to be "visible"
    outside the einsums library. Default visibility is set to hidden
    on some platforms.
*/
#if defined(EINSUMS_LIBRARY)
#    define EINSUMS_EXPORT _EXPORT
#elif defined(EINSUMS_STATIC_LIBRARY)
#    define EINSUMS_EXPORT
#else
#    define EINSUMS_EXPORT _EXPORT
#endif
