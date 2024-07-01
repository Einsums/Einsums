#pragma once

#include "einsums/_GPUCast.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/DeviceTensor.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hipsolver/internal/hipsolver-types.h>

namespace einsums {
namespace linear_algebra {

namespace detail {

namespace gpu {

template <typename T>
__global__ void dot_kernel(T *C, 
                           const T *__restrict__ A, const T *__restrict__ B, size_t elements) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t curr_index = thread_id;

    T temp;
    make_zero(temp);
    if(thread_id == 0) {
        make_zero(*C);
    }

    while (curr_index < elements) {
        temp = temp + A[curr_index] * B[curr_index];
        curr_index += kernel_size;
    }

    einsums::gpu::atomicAdd_wrap(C, temp);
}

/**
 * Internal gemm functions.
 */
EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float *alpha, const float *a,
                         int lda, const float *b, int ldb, const float *beta, float *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double *alpha, const double *a,
                         int lda, const double *b, int ldb, const double *beta, double *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipComplex *alpha,
                         const hipComplex *a, int lda, const hipComplex *b, int ldb, const hipComplex *beta, hipComplex *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipDoubleComplex *alpha,
                         const hipDoubleComplex *a, int lda, const hipDoubleComplex *b, int ldb, const hipDoubleComplex *beta,
                         hipDoubleComplex *c, int ldc);

/**
 * Internal ger functions.
 */
EINSUMS_EXPORT void ger(int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);

EINSUMS_EXPORT void ger(int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda);

EINSUMS_EXPORT void ger(int m, int n, const hipComplex *alpha, const hipComplex *x, int incx, const hipComplex *y, int incy, hipComplex *A,
                        int lda);

EINSUMS_EXPORT void ger(int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *x, int incx, const hipDoubleComplex *y,
                        int incy, hipDoubleComplex *A, int lda);

/**
 * Internal gemv functions.
 */
EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const float *alpha, const float *a, int lda, const float *x, int incx,
                         const float *beta, float *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const double *alpha, const double *a, int lda, const double *x, int incx,
                         const double *beta, double *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const hipComplex *alpha, const hipComplex *a, int lda,
                         const hipComplex *x, int incx, const hipComplex *beta, hipComplex *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
                         const hipDoubleComplex *x, int incx, const hipDoubleComplex *beta, hipDoubleComplex *y, int incy);

/**
 * Internal scale functions.
 */
EINSUMS_EXPORT void scal(int size, const float *alpha, float *x, int incx);

EINSUMS_EXPORT void scal(int size, const double *alpha, double *x, int incx);

EINSUMS_EXPORT void scal(int size, const hipComplex *alpha, hipComplex *x, int incx);

EINSUMS_EXPORT void scal(int size, const hipDoubleComplex *alpha, hipDoubleComplex *x, int incx);

/**
 * Symmetric multiplication kernels.
 */
__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const float *A, int lda, const float *B, int ldb, float *C,
                                         int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const double *A, int lda, const double *B, int ldb,
                                         double *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const hipComplex *A, int lda, const hipComplex *B, int ldb,
                                         hipComplex *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const hipDoubleComplex *A, int lda,
                                         const hipDoubleComplex *B, int ldb, hipDoubleComplex *C, int ldc);

/**
 * Internal solvers.
 */
EINSUMS_EXPORT int gesv(int n, int nrhs, float *A, int lda, int *ipiv, float *B, int ldb, float *X, int ldx);
EINSUMS_EXPORT int gesv(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb, double *X, int ldx);
EINSUMS_EXPORT int gesv(int n, int nrhs, hipComplex *A, int lda, int *ipiv, hipComplex *B, int ldb, hipComplex *X, int ldx);
EINSUMS_EXPORT int gesv(int n, int nrhs, hipDoubleComplex *A, int lda, int *ipiv, hipDoubleComplex *B, int ldb, hipDoubleComplex *X,
                        int ldx);

/**
 * Internal axpy.
 */
EINSUMS_EXPORT void axpy(int n, const float *alpha, const float *X, int incx, float *Y, int incy);
EINSUMS_EXPORT void axpy(int n, const double *alpha, const double *X, int incx, double *Y, int incy);
EINSUMS_EXPORT void axpy(int n, const hipComplex *alpha, const hipComplex *X, int incx, hipComplex *Y, int incy);
EINSUMS_EXPORT void axpy(int n, const hipDoubleComplex *alpha, const hipDoubleComplex *X, int incx, hipDoubleComplex *Y, int incy);

/**
 * Internal axpby
 */
EINSUMS_EXPORT void axpby(int n, const float *alpha, const float *X, int incx, const float *beta, float *Y, int incy);
EINSUMS_EXPORT void axpby(int n, const double *alpha, const double *X, int incx, const double *beta, double *Y, int incy);
EINSUMS_EXPORT void axpby(int n, const hipComplex *alpha, const hipComplex *X, int incx, const hipComplex *beta, hipComplex *Y, int incy);
EINSUMS_EXPORT void axpby(int n, const hipDoubleComplex *alpha, const hipDoubleComplex *X, int incx, const hipDoubleComplex *beta,
                          hipDoubleComplex *Y, int incy);

__global__ EINSUMS_EXPORT void saxpby_kernel(int n, const float *alpha, const float *X, int incx, const float *beta, float *Y, int incy);
__global__ EINSUMS_EXPORT void daxpby_kernel(int n, const double *alpha, const double *X, int incx, const double *beta, double *Y, int incy);
__global__ EINSUMS_EXPORT void caxpby_kernel(int n, const hipComplex *alpha, const hipComplex *X, int incx, const hipComplex *beta,
                                            hipComplex *Y, int incy);
__global__ EINSUMS_EXPORT void zaxpby_kernel(int n, const hipDoubleComplex *alpha, const hipDoubleComplex *X, int incx,
                                            const hipDoubleComplex *beta, hipDoubleComplex *Y, int incy);

EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float *A, int lda, float *D);
EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double *A, int lda, double *D);
} // namespace gpu

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<CType<T, Rank>, 2, T>;
    }
void gemm(const T *alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T *beta, CType<T, Rank> *C) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    int m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    // Flip the A and B matrices. Row-major vs column major.
    gpu::gemm(TransB ? HIPBLAS_OP_T : HIPBLAS_OP_N, TransA ? HIPBLAS_OP_T : HIPBLAS_OP_N, n, m, k, (dev_datatype *)alpha, B.data(), ldb,
              A.data(), lda, (dev_datatype *)beta, C->data(), ldc);
    stream_wait();
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, Rank>, Rank, T>;
        requires ::einsums::DeviceRankBasicTensor<BType<T, Rank>, Rank, T>;
    }
T dot(const AType<T, Rank> &A,
         const BType<T, Rank> &B) {
    using namespace einsums::gpu;

    using dev_datatype = std::conditional_t<
        std::is_same_v<T, std::complex<float>>, hipComplex,
        std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    __device_ptr__ dev_datatype *gpu_out;
    auto grid = block_size(A.size());
    auto num_blocks = blocks(A.size());

    hip_catch(hipMalloc((void **) &gpu_out, sizeof(T)));

    gpu::dot_kernel<<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(
        gpu_out, A.data(), B.data(), A.size());
    stream_wait();

    T out;
    hip_catch(hipMemcpy((void *) &out, (void *) gpu_out, sizeof(T), hipMemcpyDeviceToHost));

    hip_catch(hipFree((void *) gpu_out));

    return out;
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>;
    }
void ger(const T *alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    gpu::ger(X.dim(0), Y.dim(0), (dev_datatype *)alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankBasicTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(const T *alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, const T *beta, YType<T, XYRank> *y) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    int m = A.dim(1), n = A.dim(0);

    if constexpr (!TransA) {
        gpu::gemv(HIPBLAS_OP_T, m, n, (dev_datatype *)alpha, A.data(), A.stride(0), x.data(), x.stride(0), (dev_datatype *)beta,
                          y->data(), y->stride(0));
        stream_wait();
    } else {
        gpu::gemv(HIPBLAS_OP_N, m, n, (dev_datatype *)alpha, A.data(), A.stride(0), x.data(), x.stride(0), (dev_datatype *)beta,
                          y->data(), y->stride(0));
        stream_wait();
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, ARank, T>
void scale(const T *scale, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    gpu::scal(A->size(), (dev_datatype *)scale, A->data(), 1);
    stream_wait();
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<CType<T, Rank>, 2, T>;
    }
void symm_gemm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C) {
    using namespace einsums::gpu;

    if constexpr (TransA && TransB) {
        assert(B.dim(0) == A.dim(0) && A.dim(1) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else if constexpr (TransA && !TransB) {
        assert(B.dim(1) == A.dim(0) && A.dim(1) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    } else if constexpr (!TransA && TransB) {
        assert(B.dim(0) == A.dim(1) && A.dim(0) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else {
        assert(B.dim(1) == A.dim(1) && A.dim(0) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    }
    *C = T(0.0);

    einsums::linear_algebra::detail::gpu::symm_gemm<<<block_size(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)),
                                                      blocks(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)), 0, get_stream()>>>(
        TransA, TransB, A.dim(0), C->dim(0), A.data(), A.stride(0), B.data(), B.stride(0), C->data(), C->stride(0));
    stream_wait();
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<CType<T, Rank>, 2, T>;
    }
void gemm(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    // Flip the A and B matrices. Row-major vs column major.
    gemm<TransA, TransB>((T *)alpha_gpu, A, B, (T *)beta_gpu, C);

    // These can still be done async.
    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
    hip_catch(hipFreeAsync(beta_gpu, get_stream()));
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>;
        requires !std::is_pointer_v<T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));

    ger((T *)alpha_gpu, X, Y, A);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankBasicTensor<YType<T, XYRank>, 1, T>;
        requires !std::is_pointer_v<T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, (const void *)&alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, (const void *)&beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    gemv<TransA>((T *)alpha_gpu, A, x, (T *)beta_gpu, y);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
    hip_catch(hipFreeAsync(beta_gpu, get_stream()));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, ARank, T>;
        requires !std::is_pointer_v<T>;
    }
void scale(T scale_, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *scale_gpu;

    hip_catch(hipMalloc((void **)&scale_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(scale_gpu, &scale_, sizeof(dev_datatype), hipMemcpyHostToDevice));

    scale((T *)scale_gpu, A);

    hip_catch(hipFreeAsync(scale_gpu, get_stream()));
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<BType<T, BRank>, 2, T>;
    }
int gesv(AType<T, ARank> *A, BType<T, BRank> *B) {
    using namespace einsums::gpu;
    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<T, BRank>, BRank, T>) {

        if (A->num_blocks() != B->num_blocks()) {
            throw std::runtime_error("gesv: Tensors need the same number of blocks.");
        }

        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &(B->block(i)));

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;

    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {

            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &((*B)(AllT(), A->block_range(i))));

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;
    } else {

        auto n   = A->dim(0);
        auto lda = A->stride(0);
        auto ldb = B->stride(0);

        auto nrhs = B->dim(0);

        int lwork = n;
        int info;

        __device_ptr__ int *ipiv;

#ifdef __HIP_PLATFORM_NVIDIA__
        DeviceTensor<T, BRank> X = DeviceTensor<T, BRank>(einsums::detail::DEV_ONLY, B->dims());
#endif

        einsums::gpu::hip_catch(hipMallocAsync((void **)&ipiv, sizeof(int) * lwork, einsums::gpu::get_stream()));

#ifdef __HIP_PLATFORM_NVIDIA__
        info = gpu::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb, X.data(), X.stride(0));
#elif defined(__HIP_PLATFORM_AMD__)
        info = gpu::gesv(n, nrhs, A->data(), lda, ipiv, B->data(), ldb, B->data(), ldb);
#endif

        stream_wait();

        einsums::gpu::hip_catch(hipFreeAsync(ipiv, einsums::gpu::get_stream()));
        return info;
    }
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires(!std::is_pointer_v<T>);
        requires DeviceRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_stream()));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    axpy((T *)alpha_gpu, X, Y);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires DeviceRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(const T *alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    using namespace einsums::gpu;

    gpu::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
    stream_wait();
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires DeviceRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpby(const T *alpha, const XType<T, Rank> &X, const T *beta, YType<T, Rank> *Y) {
    using namespace einsums::gpu;

    einsums::linear_algebra::detail::gpu::axpby(X.dim(0) * X.stride(0), alpha, X.data(), 1, beta, Y->data(), 1);
    stream_wait();
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires(!std::is_pointer_v<T>);
        requires DeviceRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_beta_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_beta_gpu, sizeof(dev_datatype) * 2, get_stream()));

    hip_catch(hipMemcpyAsync(alpha_beta_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));
    hip_catch(hipMemcpyAsync(&(alpha_beta_gpu[1]), &beta, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    axpby((T *)&(alpha_beta_gpu[0]), X, (T *)&(alpha_beta_gpu[1]), Y);

    hip_catch(hipFreeAsync(alpha_beta_gpu, get_stream()));
}

/**
 * @brief Computes the eigenvalues and eigenvectors of a symmetric matrix.
 */
template <bool ComputeEigenvectors, template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType,
          size_t WRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<WType<T, WRank>, 1, T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W) {
    using namespace einsums::gpu;
    int lda = A->stride(0);

    int info = detail::gpu::syev(ComputeEigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER,
                                 A->dim(0), A->data(), lda, W->data());

    stream_wait();
}

namespace detail {
namespace gpu {
/**
 * @brief Copy a list of eigenvalues onto the diagonal of a matrix.
 *
 * @param out_matrix The matrix output. Only the diagonal entries are touched.
 * @param n The number of columns and the number of eigenvalues.
 * @param lda The leading dimension of the out_matrix. lda >= n.
 * @param eigs The eigenvalues to copy.
 */
template <typename T>
__global__ void eig_to_diag(T *out_matrix, int n, int lda, const T *eigs, T expo) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    // Copy to diagonal. Assume the matrix is zeroed, or at least that the user needs the off-diagonal entries.
    for (int i = thread_id; i < n; i += num_threads) {
        out_matrix[i * lda + i] = ::pow(eigs[i], expo);
    }
}
} // namespace gpu
} // namespace detail

/**
 * @brief Computes the power of a symmetric matrix.
 */
template <template <typename, size_t> typename AType, typename T, size_t ARank>
    requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>
AType<T, ARank> pow(const AType<T, ARank> &A, T expo) {
    using namespace einsums::gpu;
    using namespace einsums;
    DeviceTensor<T, 1> Evals(Dim<1>{A.dim(0)}, ::einsums::detail::DEV_ONLY);

    DeviceTensor<T, 2> Evecs(Dim<2>{A.dim(0), A.dim(1)});

    DeviceTensor<T, 2> Diag(Dim<2>{A.dim(0), A.dim(1)}, ::einsums::detail::DEV_ONLY);

    DeviceTensor<T, 2> out(Dim<2>{A.dim(0), A.dim(1)}, ::einsums::detail::DEV_ONLY);
    DeviceTensor<T, 2> temp(Dim<2>{A.dim(0), A.dim(1)}, ::einsums::detail::DEV_ONLY);

    Evecs.assign(A);

    syev(&Evecs, &Evals);

    Diag.zero();

    detail::gpu::eig_to_diag<<<dim3(32), dim3(1), 0, get_stream()>>>(Diag.data(), Diag.dim(0), Diag.stride(0), Evals.data(), expo);

    symm_gemm<false, false>(Diag, Evecs, &out);

    return out;
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums