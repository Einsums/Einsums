#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>
#include <Einsums/TypeSupport/GPUCast.hpp>

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

template <size_t Rank, typename T1, typename T2>
__global__ void dot_kernel(BiggestTypeT<T1, T2> *C, T1 const *__restrict__ A, T2 const *__restrict__ B, size_t const *__restrict__ dims,
                           size_t const *__restrict__ strides, size_t const *__restrict__ A_strides, size_t const *__restrict__ B_strides) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    BiggestTypeT<T1, T2> temp;
    make_zero(temp);
    if (thread_id == 0) {
        make_zero(*C);
    }

    size_t A_index, B_index;

    for (size_t curr_index = thread_id; curr_index < dims[0] * strides[0]; curr_index += kernel_size) {
        A_index         = 0;
        B_index         = 0;
        size_t quotient = curr_index;

        for (int i = 0; i < Rank; i++) {
            A_index += (quotient / strides[i]) * A_strides[i];
            B_index += (quotient / strides[i]) * B_strides[i];
            quotient %= strides[i];
        }

        temp = temp + A[A_index] * B[B_index];
    }

    einsums::gpu::atomicAdd_wrap(C, temp);
}

template <size_t Rank, typename T1, typename T2>
__global__ void true_dot_kernel(BiggestTypeT<T1, T2> *C, T1 const *__restrict__ A, T2 const *__restrict__ B,
                                size_t const *__restrict__ dims, size_t const *__restrict__ strides, size_t const *__restrict__ A_strides,
                                size_t const *__restrict__ B_strides) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    BiggestTypeT<T1, T2> temp;
    make_zero(temp);
    if (thread_id == 0) {
        make_zero(*C);
    }

    size_t A_index, B_index;

    for (size_t curr_index = thread_id; curr_index < dims[0] * strides[0]; curr_index += kernel_size) {
        A_index         = 0;
        B_index         = 0;
        size_t quotient = curr_index;

        for (int i = 0; i < Rank; i++) {
            A_index += (quotient / strides[i]) * A_strides[i];
            B_index += (quotient / strides[i]) * B_strides[i];
            quotient %= strides[i];
        }

        if constexpr (std::is_same_v<T1, hipComplex> || std::is_same_v<T1, hipDoubleComplex>) {
            T1 conjugate = A[curr_index];
            conjugate.y  = -conjugate.y;
            temp         = temp + conjugate * B[curr_index];
        } else {
            temp = temp + A[curr_index] * B[curr_index];
        }
    }

    einsums::gpu::atomicAdd_wrap(C, temp);
}

/**
 * Internal gemm functions.
 */
EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, float const *alpha, float const *a,
                         int lda, float const *b, int ldb, float const *beta, float *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, double const *alpha, double const *a,
                         int lda, double const *b, int ldb, double const *beta, double *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, hipComplex const *alpha,
                         hipComplex const *a, int lda, hipComplex const *b, int ldb, hipComplex const *beta, hipComplex *c, int ldc);

EINSUMS_EXPORT void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, hipDoubleComplex const *alpha,
                         hipDoubleComplex const *a, int lda, hipDoubleComplex const *b, int ldb, hipDoubleComplex const *beta,
                         hipDoubleComplex *c, int ldc);

/**
 * Internal ger functions.
 */
EINSUMS_EXPORT void ger(int m, int n, float const *alpha, float const *x, int incx, float const *y, int incy, float *A, int lda);

EINSUMS_EXPORT void ger(int m, int n, double const *alpha, double const *x, int incx, double const *y, int incy, double *A, int lda);

EINSUMS_EXPORT void ger(int m, int n, hipComplex const *alpha, hipComplex const *x, int incx, hipComplex const *y, int incy, hipComplex *A,
                        int lda);

EINSUMS_EXPORT void ger(int m, int n, hipDoubleComplex const *alpha, hipDoubleComplex const *x, int incx, hipDoubleComplex const *y,
                        int incy, hipDoubleComplex *A, int lda);

/**
 * Internal gemv functions.
 */
EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, float const *alpha, float const *a, int lda, float const *x, int incx,
                         float const *beta, float *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, double const *alpha, double const *a, int lda, double const *x, int incx,
                         double const *beta, double *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, hipComplex const *alpha, hipComplex const *a, int lda,
                         hipComplex const *x, int incx, hipComplex const *beta, hipComplex *y, int incy);

EINSUMS_EXPORT void gemv(hipblasOperation_t transa, int m, int n, hipDoubleComplex const *alpha, hipDoubleComplex const *a, int lda,
                         hipDoubleComplex const *x, int incx, hipDoubleComplex const *beta, hipDoubleComplex *y, int incy);

/**
 * Internal scale functions.
 */
EINSUMS_EXPORT void scal(int size, float const *alpha, float *x, int incx);

EINSUMS_EXPORT void scal(int size, double const *alpha, double *x, int incx);

EINSUMS_EXPORT void scal(int size, hipComplex const *alpha, hipComplex *x, int incx);

EINSUMS_EXPORT void scal(int size, hipDoubleComplex const *alpha, hipDoubleComplex *x, int incx);

/**
 * Symmetric multiplication kernels.
 */
__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, float const *A, int lda, float const *B, int ldb, float *C,
                                         int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, double const *A, int lda, double const *B, int ldb,
                                         double *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, hipComplex const *A, int lda, hipComplex const *B, int ldb,
                                         hipComplex *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, hipDoubleComplex const *A, int lda,
                                         hipDoubleComplex const *B, int ldb, hipDoubleComplex *C, int ldc);

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
EINSUMS_EXPORT void axpy(int n, float const *alpha, float const *X, int incx, float *Y, int incy);
EINSUMS_EXPORT void axpy(int n, double const *alpha, double const *X, int incx, double *Y, int incy);
EINSUMS_EXPORT void axpy(int n, hipComplex const *alpha, hipComplex const *X, int incx, hipComplex *Y, int incy);
EINSUMS_EXPORT void axpy(int n, hipDoubleComplex const *alpha, hipDoubleComplex const *X, int incx, hipDoubleComplex *Y, int incy);

/**
 * Internal axpby
 */
EINSUMS_EXPORT void axpby(int n, float const *alpha, float const *X, int incx, float const *beta, float *Y, int incy);
EINSUMS_EXPORT void axpby(int n, double const *alpha, double const *X, int incx, double const *beta, double *Y, int incy);
EINSUMS_EXPORT void axpby(int n, hipComplex const *alpha, hipComplex const *X, int incx, hipComplex const *beta, hipComplex *Y, int incy);
EINSUMS_EXPORT void axpby(int n, hipDoubleComplex const *alpha, hipDoubleComplex const *X, int incx, hipDoubleComplex const *beta,
                          hipDoubleComplex *Y, int incy);

__global__ EINSUMS_EXPORT void saxpby_kernel(int n, float const *alpha, float const *X, int incx, float const *beta, float *Y, int incy);
__global__ EINSUMS_EXPORT void daxpby_kernel(int n, double const *alpha, double const *X, int incx, double const *beta, double *Y,
                                             int incy);
__global__ EINSUMS_EXPORT void caxpby_kernel(int n, hipComplex const *alpha, hipComplex const *X, int incx, hipComplex const *beta,
                                             hipComplex *Y, int incy);
__global__ EINSUMS_EXPORT void zaxpby_kernel(int n, hipDoubleComplex const *alpha, hipDoubleComplex const *X, int incx,
                                             hipDoubleComplex const *beta, hipDoubleComplex *Y, int incy);

EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float *A, int lda, float *D);
EINSUMS_EXPORT int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double *A, int lda, double *D);

template <size_t Rank, typename T>
__global__ void direct_product_kernel(T beta, T *__restrict__ C, size_t const *__restrict__ C_strides, T alpha, T const *__restrict__ A,
                                      size_t const *__restrict__ A_strides, T const *__restrict__ B, size_t const *__restrict__ B_strides,
                                      size_t const *__restrict__ Index_dims, size_t const *__restrict__ Index_strides, size_t elems) {
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t index[Rank];

    for (size_t curr = thread_id; curr < elems; curr += kernel_size) {

        einsums::sentinel_to_indices<Rank>(curr, Index_strides, index);

        size_t A_index = 0, B_index = 0, C_index = 0;

        for (int i = 0; i < Rank; i++) {
            A_index += index[i] * A_strides[i];
            B_index += index[i] * B_strides[i];
            C_index += index[i] * C_strides[i];
        }

        if (beta == T{0.0}) {
            C[C_index] = alpha * A[A_index] * B[B_index];
        } else {
            C[C_index] = beta * C[C_index] + alpha * A[A_index] * B[B_index];
        }
    }
}

} // namespace gpu

template <bool TransA, bool TransB, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType,
          typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::is_same_v<typename AType::host_datatype, T>;
    }
void gemm(T const *alpha, AType const &A, BType const &B, T const *beta, CType *C) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    int m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    // Flip the A and B matrices. Row-major vs column major.
    gpu::gemm(TransB ? HIPBLAS_OP_T : HIPBLAS_OP_N, TransA ? HIPBLAS_OP_T : HIPBLAS_OP_N, n, m, k, (dev_datatype *)alpha, B.gpu_data(), ldb,
              A.gpu_data(), lda, (dev_datatype *)beta, C->gpu_data(), ldc);
    stream_wait();
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType>
    requires(SameRank<AType, BType>)
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    using namespace einsums::gpu;

    using dev_datatype    = typename AType::dev_datatype;
    using T               = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;
    constexpr size_t Rank = AType::Rank;

    dev_datatype            *gpu_out;
    auto                     grid       = block_size(A.size());
    auto                     num_blocks = blocks(A.size());
    size_t                  *gpu_strides;
    std::array<size_t, Rank> strides;

    hip_catch(hipMalloc((void **)&gpu_out, sizeof(T)));
    hip_catch(hipMalloc((void **)&gpu_strides, Rank * sizeof(size_t)));

    size_t prod = 1;

    for (int i = Rank - 1; i >= 0; i--) {
        strides[i] = prod;
        prod *= A.dim(i);
    }

    hip_catch(hipMemcpy(gpu_strides, strides.data(), Rank * sizeof(size_t), hipMemcpyHostToDevice));

    gpu::dot_kernel<Rank><<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(gpu_out, A.gpu_data(), B.gpu_data(), A.gpu_dims(),
                                                                                       gpu_strides, A.gpu_strides(), B.gpu_strides());
    stream_wait();

    T out;
    hip_catch(hipMemcpy((void *)&out, (void *)gpu_out, sizeof(T), hipMemcpyDeviceToHost));
    // No sync

    hip_catch(hipFree((void *)gpu_out));
    hip_catch(hipFree((void *)gpu_strides));

    return out;
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType>
    requires(SameRank<AType, BType>)
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    using namespace einsums::gpu;

    using dev_datatype    = typename AType::dev_datatype;
    using T               = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;
    constexpr size_t Rank = AType::Rank;

    dev_datatype            *gpu_out;
    size_t                  *gpu_strides;
    std::array<size_t, Rank> strides;

    auto grid       = block_size(A.size());
    auto num_blocks = blocks(A.size());

    hip_catch(hipMalloc((void **)&gpu_out, sizeof(T)));
    hip_catch(hipMalloc((void **)&gpu_strides, Rank * sizeof(size_t)));

    size_t prod = 1;

    for (int i = Rank - 1; i >= 0; i--) {
        strides[i] = prod;
        prod *= A.dim(i);
    }

    hip_catch(hipMemcpy(gpu_strides, strides.data(), Rank * sizeof(size_t), hipMemcpyHostToDevice));

    gpu::true_dot_kernel<Rank><<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(
        gpu_out, A.gpu_data(), B.gpu_data(), A.gpu_dims(), gpu_strides, A.gpu_strides(), B.gpu_strides());
    stream_wait();

    T out;
    hip_catch(hipMemcpy((void *)&out, (void *)gpu_out, sizeof(T), hipMemcpyDeviceToHost));
    // No sync

    hip_catch(hipFree((void *)gpu_out));
    hip_catch(hipFree((void *)gpu_strides));

    return out;
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void ger(T const *alpha, XType const &X, YType const &Y, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    gpu::ger(Y.dim(0), X.dim(0), (dev_datatype *)alpha, Y.gpu_data(), Y.stride(0), X.gpu_data(), X.stride(0), A->gpu_data(), A->stride(0));

    // No wait needed. sort waits.
}

template <bool TransA, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void gemv(T const *alpha, AType const &A, XType const &x, T const *beta, YType *y) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    int m = A.dim(1), n = A.dim(0);

    if constexpr (!TransA) {
        gpu::gemv(HIPBLAS_OP_T, m, n, (dev_datatype *)alpha, A.gpu_data(), A.stride(0), x.gpu_data(), x.stride(0), (dev_datatype *)beta,
                  y->gpu_data(), y->stride(0));
        stream_wait();
    } else {
        gpu::gemv(HIPBLAS_OP_N, m, n, (dev_datatype *)alpha, A.gpu_data(), A.stride(0), x.gpu_data(), x.stride(0), (dev_datatype *)beta,
                  y->gpu_data(), y->stride(0));
        stream_wait();
    }
}

template <DeviceBasicTensorConcept AType, typename T>
    requires(std::is_same_v<typename AType::ValueType, T>)
void scale(T const *alpha, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    // TODO: Compatibility with views.
    gpu::scal(A->size(), (dev_datatype *)alpha, A->gpu_data(), 1);
    stream_wait();
}

template <bool TransA, bool TransB, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
    }
void symm_gemm(AType const &A, BType const &B, CType *C) {
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
    *C = typename CType::ValueType(0.0);

    einsums::linear_algebra::detail::gpu::symm_gemm<<<block_size(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)),
                                                      blocks(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)), 0, get_stream()>>>(
        TransA, TransB, A.dim(0), C->dim(0), A.gpu_data(), A.stride(0), B.gpu_data(), B.stride(0), C->gpu_data(), C->stride(0));
    stream_wait();
}

template <bool TransA, bool TransB, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType,
          typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void gemm(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;
    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpyAsync((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));
    hip_catch(hipMemcpyAsync((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    gemm<TransA, TransB>((T *)alpha_gpu, A, B, (T *)beta_gpu, C);

    // These can still be done async.
    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
    hip_catch(hipFreeAsync(beta_gpu, get_stream()));
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void ger(T alpha, XType const &X, YType const &Y, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::ValueType;

    dev_datatype *alpha_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    ger((T *)alpha_gpu, X, Y, A);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <bool TransA, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void gemv(T alpha, AType const &A, XType const &x, T beta, YType *y) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;
    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpyAsync((void *)alpha_gpu, (void const *)&alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));
    hip_catch(hipMemcpyAsync((void *)beta_gpu, (void const *)&beta, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    gemv<TransA>((T *)alpha_gpu, A, x, (T *)beta_gpu, y);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
    hip_catch(hipFreeAsync(beta_gpu, get_stream()));
}

template <DeviceBasicTensorConcept AType, typename T>
    requires(std::is_same_v<typename AType::ValueType, T>)
void scale(T scale_, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    dev_datatype *scale_gpu;

    hip_catch(hipMalloc((void **)&scale_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpyAsync(scale_gpu, &scale_, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    scale((T *)scale_gpu, A);

    hip_catch(hipFreeAsync(scale_gpu, get_stream()));
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType>;
    }
int gesv(AType *A, BType *B) {
    using namespace einsums::gpu;
    auto n   = A->dim(0);
    auto lda = A->stride(0);
    auto ldb = B->stride(0);

    auto nrhs = B->dim(0);

    int lwork = n;
    int info;

    int *ipiv;

#ifdef __HIP_PLATFORM_NVIDIA__
    DeviceTensor<typename AType::ValueType, 2> X = DeviceTensor<typename AType::ValueType, 2>(einsums::detail::DEV_ONLY, B->dims());
#endif

    einsums::hip_catch(hipMallocAsync((void **)&ipiv, sizeof(int) * lwork, einsums::gpu::get_stream()));

#ifdef __HIP_PLATFORM_NVIDIA__
    info = gpu::gesv(n, nrhs, A->gpu_data(), lda, ipiv, B->gpu_data(), ldb, X.gpu_data(), X.stride(0));
#elif defined(__HIP_PLATFORM_AMD__)
    info = gpu::gesv(n, nrhs, A->gpu_data(), lda, ipiv, B->gpu_data(), ldb, B->gpu_data(), ldb);
#endif

    stream_wait();

    einsums::hip_catch(hipFreeAsync(ipiv, einsums::gpu::get_stream()));
    return info;
}

template <DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<XType, YType>;
        requires std::is_same_v<typename XType::ValueType, T>;
    }
void axpy(T const *alpha, XType const &X, YType *Y) {
    using namespace einsums::gpu;

    gpu::axpy(X.dim(0) * X.stride(0), alpha, X.gpu_data(), 1, Y->gpu_data(), 1);
    stream_wait();
}

template <DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<XType, YType>;
        requires std::is_same_v<typename XType::ValueType, T>;
    }
void axpby(T const *alpha, XType const &X, T const *beta, YType *Y) {
    using namespace einsums::gpu;

    einsums::linear_algebra::detail::gpu::axpby(X.dim(0) * X.stride(0), alpha, X.gpu_data(), 1, beta, Y->gpu_data(), 1);
    stream_wait();
}

template <DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<XType, YType>;
        requires std::is_same_v<typename XType::ValueType, T>;
    }
void axpy(T alpha, XType const &X, YType *Y) {
    using namespace einsums::gpu;

    using dev_datatype = typename XType::dev_datatype;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_stream()));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    axpy((T *)alpha_gpu, X, Y);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<XType, YType>;
        requires std::is_same_v<typename XType::ValueType, T>;
    }
void axpby(T alpha, XType const &X, T beta, YType *Y) {
    using namespace einsums::gpu;

    using dev_datatype = typename XType::dev_datatype;

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

template <bool ComputeEigenvectors = true, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept WType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<WType>;
        requires SameUnderlying<AType, WType>;
    }
void syev(AType *A, WType *W) {
    using namespace einsums::gpu;
    int lda = A->stride(0);

    int info = detail::gpu::syev(ComputeEigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER,
                                 A->dim(0), A->gpu_data(), lda, W->gpu_data());

    stream_wait();
}

template <DeviceBasicTensorConcept AType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void scale_row(size_t row, T const *alpha, AType *A) {
    using namespace einsums::gpu;
    gpu::scal(A->dim(1), alpha, A->gpu_data(row, 0ul), A->stride(1));
    stream_wait();
}

template <DeviceBasicTensorConcept AType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void scale_column(size_t col, T const *alpha, AType *A) {
    using namespace einsums::gpu;
    gpu::scal(A->dim(0), alpha, A->gpu_data(0ul, col), A->stride(0));
    stream_wait();
}

template <DeviceBasicTensorConcept AType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void scale_row(size_t row, T alpha, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_stream()));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    scale_row(row, (T *)alpha_gpu, A);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <DeviceBasicTensorConcept AType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void scale_column(size_t col, T alpha, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_stream()));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    scale_column(col, (T *)alpha_gpu, A);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void direct_product(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    using namespace einsums::gpu;

    using T_devtype  = typename CType::dev_datatype;
    using T_hosttype = typename CType::host_datatype;

    constexpr size_t Rank = CType::Rank;

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
        HipCast<T_devtype, T_hosttype>::cast(beta), C->gpu_data(), C->gpu_strides(), HipCast<T_devtype, T_hosttype>::cast(alpha),
        A.gpu_data(), A.gpu_strides(), B.gpu_data(), B.gpu_strides(), A.gpu_dims(), gpu_Ind_strides, elems);

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
template <typename T, typename U>
__global__ void eig_to_diag(T *__restrict__ out_matrix, int n, int lda, U const *__restrict__ eigs, T expo) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    // Copy to diagonal. Assume the matrix is zeroed, or at least that the user needs the off-diagonal entries.
    for (int i = thread_id; i < n; i += num_threads) {
        out_matrix[i * lda + i] = ::pow(HipCast<T, U>::cast(eigs[i]), expo);
    }
}
} // namespace gpu

template <MatrixConcept AType>
    requires DeviceBasicTensorConcept<AType>
auto pow(AType const &a, typename AType::host_datatype alpha,
         typename AType::host_datatype cutoff = std::numeric_limits<typename AType::host_datatype>::epsilon()) -> RemoveViewT<AType> {
    using T = typename AType::host_datatype;
    DeviceTensor<RemoveComplexT<T>, 1> Evals(Dim<1>{a.dim(0)}, ::einsums::detail::DEV_ONLY);

    RemoveViewT<AType> Evecs = create_tensor_like(a);

    RemoveViewT<AType> Diag = create_tensor_like(a);

    RemoveViewT<AType> out  = create_tensor_like(a);
    RemoveViewT<AType> temp = create_tensor_like(a);

    Evecs.assign(a);

    if constexpr (einsums::IsComplexV<AType>) {
        hyev<true>(&Evecs, &Evals);
    } else {
        syev<true>(&Evecs, &Evals);
    }

    Diag.zero();

    gpu::eig_to_diag<<<dim3(32), dim3(1), 0, einsums::gpu::get_stream()>>>(Diag.gpu_data(), Diag.dim(0), Diag.stride(0), Evals.gpu_data(),
                                                                           alpha);

    symm_gemm<false, false>(Diag, Evecs, &out);

    return out;
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums