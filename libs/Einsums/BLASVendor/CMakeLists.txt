#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(EINSUMS_LINALG_VENDOR STREQUAL MKL)
  einsums_add_config_define_namespace(DEFINE EINSUMS_HAVE_MKL NAMESPACE BLASVENDOR)
endif()

set(BLASVendorHeaders Einsums/BLASVendor/Vendor.hpp)

set(BLASVendorSources
    InitializeFinalize.cpp
    axpy.cpp
    copy.cpp
    dot.cpp
    gees.cpp
    geev.cpp
    gemm.cpp
    gemv.cpp
    geqrf.cpp
    ger.cpp
    gesv.cpp
    gesdd.cpp
    gesvd.cpp
    getri.cpp
    getrf.cpp
    heev.cpp
    lange.cpp
    lassq.cpp
    orgqr.cpp
    scal.cpp
    syev.cpp
    trsyl.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums BLASVendor
  SOURCES ${BLASVendorSources}
  HEADERS ${BLASVendorHeaders}
  DEPENDENCIES tgt::lapack
        MODULE_DEPENDENCIES Einsums_Assertion Einsums_Config Einsums_Print Einsums_Profile
        Einsums_BLASBase Einsums_HPTT Einsums_Errors Einsums_Logging Einsums_BufferAllocator
  CMAKE_SUBDIRS examples tests
)
