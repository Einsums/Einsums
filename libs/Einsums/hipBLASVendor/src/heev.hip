//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Defines.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <exception>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>

#include "Common.hpp"

namespace einsums::blas::hip {

auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork) -> int {
    LabeledSection0();

    if (lwork == -1) {
        int lwork_out;
        hipsolver_catch(hipsolverCheevd_bufferSize(
            gpu::get_blas_handle(), (vendor::lsame(job, 'v') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR),
            (vendor::lsame(uplo, 'l') ? HIPSOLVER_FILL_MODE_LOWER : HIPSOLVER_FILL_MODE_UPPER), n, (hipComplex *)a, lda, w, &lwork_out));
        gpu::stream_wait();
        return lwork_out;
    } else {
        int info;

        int *dev_info = gpu::register_host_variable(info);

        try {
            hipsolver_catch(hipsolverCheevd(gpu::get_blas_handle(),
                                            (vendor::lsame(job, 'v') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR),
                                            (vendor::lsame(uplo, 'l') ? HIPSOLVER_FILL_MODE_LOWER : HIPSOLVER_FILL_MODE_UPPER), n,
                                            (hipComplex *)a, lda, w, (hipComplex *)work, lwork, dev_info));
            gpu::stream_wait();
        } catch (std::exception &) {
            gpu::unregister_host_variable(info);
            std::rethrow_exception(std::current_exception());
        }
        gpu::unregister_host_variable(info);
        return info;
    }
}
auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork) -> int {
    LabeledSection0();

    if (lwork == -1) {
        int lwork_out;
        hipsolver_catch(hipsolverZheevd_bufferSize(gpu::get_blas_handle(),
                                                   (vendor::lsame(job, 'v') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR),
                                                   (vendor::lsame(uplo, 'l') ? HIPSOLVER_FILL_MODE_LOWER : HIPSOLVER_FILL_MODE_UPPER), n,
                                                   (hipDoubleComplex *)a, lda, w, &lwork_out));
        gpu::stream_wait();
        return lwork_out;
    } else {
        int info;

        int *dev_info = gpu::register_host_variable(info);

        try {
            hipsolver_catch(hipsolverZheevd(gpu::get_blas_handle(),
                                            (vendor::lsame(job, 'v') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR),
                                            (vendor::lsame(uplo, 'l') ? HIPSOLVER_FILL_MODE_LOWER : HIPSOLVER_FILL_MODE_UPPER), n,
                                            (hipDoubleComplex *)a, lda, w, (hipDoubleComplex *)work, lwork, dev_info));
            gpu::stream_wait();
        } catch (std::exception &) {
            gpu::unregister_host_variable(info);
            std::rethrow_exception(std::current_exception());
        }
        gpu::unregister_host_variable(info);
        return info;
    }
}

} // namespace einsums::blas::hip