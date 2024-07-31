#pragma once

#include "einsums/_GPUCast.hpp"
#include "einsums/_GPUUtils.hpp"
#include "einsums/_Index.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hipsolver/internal/hipsolver-types.h>

namespace einsums::tensor_algebra {
namespace detail {}
template <NotASmartPointer ObjectA, NotASmartPointer ObjectC, typename... CIndices, typename... AIndices>
void sort(const std::tuple<CIndices...> &C_indices, ObjectC *C, const std::tuple<AIndices...> &A_indices, const ObjectA &A);
} // namespace einsums::tensor_algebra

namespace einsums {
namespace linear_algebra {

namespace detail {

namespace gpu {

template <typename T>
__global__ void dot_kernel(T *C, const T *__restrict__ A, const T *__restrict__ B, size_t elements) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t curr_index = thread_id;

    T temp;
    make_zero(temp);
    if (thread_id == 0) {
        make_zero(*C);
    }

    while (curr_index < elements) {
        temp = temp + A[curr_index] * B[curr_index];
        curr_index += kernel_size;
    }

    einsums::gpu::atomicAdd_wrap(C, temp);
}

template <typename T>
__global__ void true_dot_kernel(T *C, const T *__restrict__ A, const T *__restrict__ B, size_t elements) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t curr_index = thread_id;

    T temp;
    make_zero(temp);
    if (thread_id == 0) {
        make_zero(*C);
    }

    while (curr_index < elements) {
        if constexpr (std::is_same_v<T, hipComplex> || std::is_same_v<T, hipDoubleComplex>) {
            T conjugate = A[curr_index];
            conjugate.y = -conjugate.y;
            temp        = temp + conjugate * B[curr_index];
        } else {
            temp = temp + A[curr_index] * B[curr_index];
        }
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
__global__ EINSUMS_EXPORT void daxpby_kernel(int n, const double *alpha, const double *X, int incx, const double *beta, double *Y,
                                             int incy);
__global__ EINSUMS_EXPORT void caxpby_kernel(int n, const hipComplex *alpha, const hipComplex *X, int incx, const hipComplex *beta,
                                             hipComplex *Y, int incy);
__global__ EINSUMS_EXPORT void zaxpby_kernel(int n, const hipDoubleComplex *alpha, const hipDoubleComplex *X, int incx,
                                             const hipDoubleComplex *beta, hipDoubleComplex *Y, int incy);

EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float *A, int lda, float *D);
EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double *A, int lda, double *D);

template <size_t Rank, typename T>
__global__ void direct_product_kernel(T beta, T *__restrict__ C, const size_t *__restrict__ C_strides, T alpha, const T *__restrict__ A,
                                      const size_t *__restrict__ A_strides, const T *__restrict__ B, const size_t *__restrict__ B_strides,
                                      const size_t *__restrict__ Index_dims, const size_t *__restrict__ Index_strides, size_t elems) {
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t index[Rank];

    for (size_t curr = thread_id; curr < elems; curr += kernel_size) {

        einsums::tensor_algebra::detail::sentinel_to_indices<Rank>(curr, Index_strides, index);

        size_t A_index = 0, B_index = 0, C_index = 0;

        for (int i = 0; i < Rank; i++) {
            A_index += index[i] * A_strides[i];
            B_index += index[i] * B_strides[i];
            C_index += index[i] * C_strides[i];
        }

        C[C_index] = beta * C[C_index] + alpha * A[A_index] * B[B_index];
    }
}

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
    gpu::gemm(TransB ? HIPBLAS_OP_T : HIPBLAS_OP_N, TransA ? HIPBLAS_OP_T : HIPBLAS_OP_N, n, m, k, (dev_datatype *)alpha, B.gpu_data(), ldb,
              A.gpu_data(), lda, (dev_datatype *)beta, C->gpu_data(), ldc);
    stream_wait();
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, Rank>, Rank, T>;
        requires ::einsums::DeviceRankBasicTensor<BType<T, Rank>, Rank, T>;
    }
T dot(const AType<T, Rank> &A, const BType<T, Rank> &B) {
    using namespace einsums::gpu;

    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    __device_ptr__ dev_datatype *gpu_out;
    auto                         grid       = block_size(A.size());
    auto                         num_blocks = blocks(A.size());

    hip_catch(hipMalloc((void **)&gpu_out, sizeof(T)));

    gpu::dot_kernel<<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(gpu_out, A.gpu_data(), B.gpu_data(), A.size());
    stream_wait();

    T out;
    hip_catch(hipMemcpy((void *)&out, (void *)gpu_out, sizeof(T), hipMemcpyDeviceToHost));

    hip_catch(hipFree((void *)gpu_out));

    return out;
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, Rank>, Rank, T>;
        requires ::einsums::DeviceRankBasicTensor<BType<T, Rank>, Rank, T>;
    }
T true_dot(const AType<T, Rank> &A, const BType<T, Rank> &B) {
    using namespace einsums::gpu;

    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    __device_ptr__ dev_datatype *gpu_out;
    auto                         grid       = block_size(A.size());
    auto                         num_blocks = blocks(A.size());

    hip_catch(hipMalloc((void **)&gpu_out, sizeof(T)));

    gpu::true_dot_kernel<<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(gpu_out, A.gpu_data(), B.gpu_data(), A.size());
    stream_wait();

    T out;
    hip_catch(hipMemcpy((void *)&out, (void *)gpu_out, sizeof(T), hipMemcpyDeviceToHost));

    hip_catch(hipFree((void *)gpu_out));

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

    DeviceTensor<T, ARank> A_temp{"ger temp", einsums::detail::DEV_ONLY, A->dim(1), A->dim(0)};

    einsums::tensor_algebra::sort(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i, einsums::tensor_algebra::index::j}, &A_temp,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j, einsums::tensor_algebra::index::i}, *A);

    gpu::ger(X.dim(0), Y.dim(0), (dev_datatype *)alpha, X.gpu_data(), X.stride(0), Y.gpu_data(), Y.stride(0), A_temp.gpu_data(), A_temp.stride(0));

    einsums::tensor_algebra::sort(einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i, einsums::tensor_algebra::index::j}, A,
                                  einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j, einsums::tensor_algebra::index::i},
                                  A_temp);

    // No wait needed. sort waits.
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
        gpu::gemv(HIPBLAS_OP_T, m, n, (dev_datatype *)alpha, A.gpu_data(), A.stride(0), x.gpu_data(), x.stride(0), (dev_datatype *)beta, y->gpu_data(),
                  y->stride(0));
        stream_wait();
    } else {
        gpu::gemv(HIPBLAS_OP_N, m, n, (dev_datatype *)alpha, A.gpu_data(), A.stride(0), x.gpu_data(), x.stride(0), (dev_datatype *)beta, y->gpu_data(),
                  y->stride(0));
        stream_wait();
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, ARank, T>
void scale(const T *alpha, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    gpu::scal(A->size(), (dev_datatype *)alpha, A->gpu_data(), 1);
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
        TransA, TransB, A.dim(0), C->dim(0), A.gpu_data(), A.stride(0), B.gpu_data(), B.stride(0), C->gpu_data(), C->stride(0));
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
    info = gpu::gesv(n, nrhs, A->gpu_data(), lda, ipiv.gpu_data(), B->gpu_data(), ldb, X.gpu_data(), X.stride(0));
#elif defined(__HIP_PLATFORM_AMD__)
    info = gpu::gesv(n, nrhs, A->gpu_data(), lda, ipiv, B->gpu_data(), ldb, B->gpu_data(), ldb);
#endif

    stream_wait();

    einsums::gpu::hip_catch(hipFreeAsync(ipiv, einsums::gpu::get_stream()));
    return info;
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires DeviceRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(const T *alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    using namespace einsums::gpu;

    gpu::axpy(X.dim(0) * X.stride(0), alpha, X.gpu_data(), 1, Y->gpu_data(), 1);
    stream_wait();
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires DeviceRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpby(const T *alpha, const XType<T, Rank> &X, const T *beta, YType<T, Rank> *Y) {
    using namespace einsums::gpu;

    einsums::linear_algebra::detail::gpu::axpby(X.dim(0) * X.stride(0), alpha, X.gpu_data(), 1, beta, Y->gpu_data(), 1);
    stream_wait();
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
template <bool ComputeEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <typename, size_t> typename WType, size_t WRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankBasicTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankBasicTensor<WType<T, WRank>, 1, T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W) {
    using namespace einsums::gpu;
    int lda = A->stride(0);

    int info = detail::gpu::syev(ComputeEigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER,
                                 A->dim(0), A->gpu_data(), lda, W->gpu_data());

    stream_wait();
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires DeviceRankBasicTensor<AType<T, ARank>, 2, T>
void scale_row(size_t row, const T *alpha, AType<T, ARank> *A) {
    gpu::scal(A->dim(1), alpha, A->gpu_data(row, 0ul), A->stride(1));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires DeviceRankBasicTensor<AType<T, ARank>, 2, T>
void scale_column(size_t col, const T *alpha, AType<T, ARank> *A) {
    gpu::scal(A->dim(0), alpha, A->gpu_data(0ul, col), A->stride(0));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires DeviceRankBasicTensor<AType<T, ARank>, 2, T>
void scale_row(size_t row, T alpha, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_stream()));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    scale_row(row, (T *)alpha_gpu, A);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires DeviceRankBasicTensor<AType<T, ARank>, 2, T>
void scale_column(size_t col, T alpha, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_stream()));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    scale_column(col, (T *)alpha_gpu, A);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires DeviceRankTensor<AType<T, Rank>, Rank, T>;
        requires DeviceRankTensor<BType<T, Rank>, Rank, T>;
        requires DeviceRankTensor<CType<T, Rank>, Rank, T>;
    }
void direct_product(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    using namespace einsums::gpu;

    using T_devtype  = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<decltype(C->gpu_data())>>>;
    using T_hosttype = std::remove_cvref_t<std::remove_pointer_t<std::decay_t<T>>>;

    assert(A.dims() == B.dims() && A.dims() == C->dims());

    size_t elems = A.size();

    std::array<size_t, Rank> Index_strides;

    size_t prod = 1;

    for (int i = Rank - 1; i >= 0; i--) {
        Index_strides[i] = prod;
        prod *= A.dim(i);
    }

    size_t *gpu_Ind_strides;

    hip_catch(hipMalloc((void **)&gpu_Ind_strides, Rank * sizeof(size_t)));

    hip_catch(hipMemcpy((void *)gpu_Ind_strides, Index_strides.data(), Rank * sizeof(size_t), hipMemcpyHostToDevice));

    dim3 threads = block_size(elems), num_blocks = blocks(elems);

    gpu::direct_product_kernel<Rank><<<threads, num_blocks, 0, get_stream()>>>(
        HipCast<T_devtype, T_hosttype>::cast(beta), C->gpu_data(), C->gpu_strides(), HipCast<T_devtype, T_hosttype>::cast(alpha), A.gpu_data(),
        A.gpu_strides(), B.gpu_data(), B.gpu_strides(), A.gpu_dims(), gpu_Ind_strides, elems);

    stream_wait();

    hip_catch(hipFree((void *)gpu_Ind_strides));
}

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

template <MatrixConcept AType>
    requires DeviceBasicTensorConcept<AType>
auto pow(const AType &a, typename AType::host_datatype alpha,
         typename AType::host_datatype cutoff = std::numeric_limits<typename AType::host_datatype>::epsilon()) -> remove_view_t<AType> {
    using T = typename AType::host_datatype;
    DeviceTensor<T, 1> Evals(Dim<1>{a.dim(0)}, ::einsums::detail::DEV_ONLY);

    remove_view_t<AType> Evecs = create_tensor_like(a);

    remove_view_t<AType> Diag = create_tensor_like(a);

    remove_view_t<AType> out  = create_tensor_like(a);
    remove_view_t<AType> temp = create_tensor_like(a);

    Evecs.assign(a);

    syev<true>(&Evecs, &Evals);

    Diag.zero();

    gpu::eig_to_diag<<<dim3(32), dim3(1), 0, einsums::gpu::get_stream()>>>(Diag.gpu_data(), Diag.dim(0), Diag.stride(0), Evals.gpu_data(), alpha);

    symm_gemm<false, false>(Diag, Evecs, &out);

    return out;
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums