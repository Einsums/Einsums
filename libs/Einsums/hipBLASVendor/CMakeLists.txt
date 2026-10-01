#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_GPU_SUPPORT)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(hipBLASVendorHeaders Einsums/hipBLASVendor/Vendor.hpp)

set(hipBLASVendorSources
    asum.hip
    axpy.hip
    copy.hip
    dot.hip
    # gees.hip
    # geev.hip
    # gelqf.hip
    gemm.hip
    gemv.hip
    geqrf.hip
    ger.hip
    gesv.hip
    # gesdd.hip
    gesvd.hip
    getri.hip
    getrf.hip
    heev.hip
    lacgv.hip
    # lange.hip
    lassq.hip
    # orglq.hip
    orgqr.hip
    scal.hip
    syev.hip
    # trsyl.hip
    direct_prod.hip
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums hipBLASVendor
  SOURCES ${hipBLASVendorSources}
  HEADERS ${hipBLASVendorHeaders}
  DEPENDENCIES hipblas hipsolver hip::host hip::device
        MODULE_DEPENDENCIES Einsums_Assertion Einsums_Config Einsums_Print Einsums_Profile
                            Einsums_BLASBase Einsums_HPTT Einsums_Errors Einsums_Logging Einsums_BufferAllocator
                            Einsums_GPUStreams Einsums_GPUMemory
  CMAKE_SUBDIRS examples tests
)

endif()