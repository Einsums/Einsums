..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_BLASVendor:

==========
hipBLASVendor
==========

This module contains wrappings for hipBLAS and hipSOLVER that are used by Einsums. They are considered
internal and are used by higher level calls.

See the :ref:`API reference <modules_Einsums_hipBLASVendor_api>` of this module for more
details.

Functions Present
-----------------

Here is a list of the function that are present:

.. versionadded:: 2.0.0
    * sasum, dasum, casum, zasum
    * saxpy, daxpy, caxpy, zaxpy
    * saxpby, daxpby, caxpby, zaxpby
    * scopy, dcopy, ccopy, zcopy
    * sdirprod, ddirprod, cdirprod, zdirprod
        * The dirprod functions are unique to Einsums on both CPU and GPU.
    * sdot, ddot, cdot, zdot
        * cdot and zdot are wrappers around `hipblasCdotu` and `hipblasZdotu`.
    * cdotc, zdotc
    * sgemm, dgemm, cgemm, zgemm
    * sgemv, dgemv, cgemv, zgemv
    * sgeqrf, dgeqrf, cgeqrf, zgeqrf
    * sger, dger, cger, zger
        * cger and zger are wrappers around `hipblasCgeru` and `hipblasZgeru`.
    * cgerc, zgerc
    * sgesv, dgesv, cgesv, zgesv
    * sgesvd, dgesvd, cgesvd, zgesvd
    * sgetrf, dgetrf, cgetrf, zgetrf
    * sgetri, dgetri, cgetri, zgetri
        * hipBLAS only has a batched version of these, so the functionality is hacked in.
    * cheev, zheev
    * clacv, zlacv
        * These have been implemented by the Einsums devs.
    * slascl, dlascl
        * These have been implemented by the Einsums devs.
    * slassq, dlassq, classq, zlassq
        * These have been implemented by the Einsums devs.
    * snrm2, dnrm2, scnrm2, dznrm2
    * sorgqr, dorgqr
    * srscl, drscl, scrscl, zdrscl
        * These have been implemented by the Einsums devs.
    * cungqr, zungqr
    * sscal, dscal, cscal, zscal, csscal, zdscal
    * scsum1, dzsum1
        * These have been implemented by the Einsums devs.
    * ssyev, dsyev
    


Functions Missing
-----------------

Several functions are not yet implemented in hipBLAS or hipSolver. Here are the ones that have not been implemented, either by AMD 
or the Einsums devs.

* sgees, dgees, cgees, zgees
* sgeev, dgeev, cgeev, zgeev
    * For some reason, hipSolver does not allow general eigendecomposition. This may change in the future.
      The algorithm is also complicated enough that it would be unreasonable to maintain our own version.
* sgelqf, dgelqf, cgelqf, zgelqf
* sgesdd, dgesdd, cgesdd, zgesdd
    * This is just an alternate algorithm for `gesvd`.
* slange, dlange, clange, zlange
* strsyl, dtrsyl, ctrsyl, ztrsyl