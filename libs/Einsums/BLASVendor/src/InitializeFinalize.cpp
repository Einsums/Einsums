//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>

#if defined(EINSUMS_HAVE_MKL)
typedef void (*XerblaEntry)(char const *Name, int const *Num, intLen const);
extern "C" {
XerblaEntry mkl_set_xerbla(XerblaEntry xerbla);
}
#endif

namespace einsums::blas::vendor {

namespace {
extern "C" void xerbla(char const *srname, int const *info, int const /*len*/) {
    if (*info == 1001) {
        println_abort("BLAS/LAPACK: Incompatible optional parameters on entry to {}", srname);
    } else if (*info == 1000 || *info == 1089) {
        println_abort("BLAS/LAPACK: Insufficient workspace available in function {}.", srname);
    } else if (*info < 0) {
        println_abort("BLAS/LAPACK: Condition {} detected in function {}.", -(*info), srname);
    } else {
        println_abort("BLAS/LAPACK: The value of parameter {} is invalid in function call to {}.", *info, srname);
    }
}

} // namespace

void initialize() {
#if defined(EINSUMS_HAVE_MKL)
    mkl_set_xerbla(&xerbla);
#endif
}

void finalize() {
}

} // namespace einsums::blas::vendor