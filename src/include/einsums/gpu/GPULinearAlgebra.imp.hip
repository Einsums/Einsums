#pragma once

#include "einsums/_GPUUtils.hpp"

#include "einsums/../../backends/linear_algebra/hipblas/hipblas.hpp"
#include "einsums/GPULinearAlgebra.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

namespace einsums {
namespace gpu {
namespace linear_algebra {

using namespace einsums::gpu::detail;
using namespace einsums::backend::linear_algebra::hipblas;
using namespace einsums::backend::linear_algebra::hipblas::detail;

namespace detail {

template <typename CDataType, typename ADataType, typename BDataType>
__global__ void dot_kernel(CDataType *C, std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor,
                           const ADataType *A, const BDataType *B, size_t elements) {
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t curr_index = thread_id;

    while (curr_index < elements) {
        atomicAdd(C, (CDataType)(AB_prefactor * A[curr_index] * B[curr_index]));
        curr_index += kernel_size;
    }
}

template <>
void gemm<float>(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float *alpha, const float *a, int lda,
                 const float *b, int ldb, const float *beta, float *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));

    hip_catch(hipDeviceSynchronize());
}

template <>
void gemm<double>(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double *alpha, const double *a, int lda,
                  const double *b, int ldb, const double *beta, double *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));

    hip_catch(hipDeviceSynchronize());
}

template <>
void gemm<hipComplex>(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipComplex *alpha,
                      const hipComplex *a, int lda, const hipComplex *b, int ldb, const hipComplex *beta, hipComplex *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgemm(handle, transa, transb, m, n, k, (const hipblasComplex *)alpha, (const hipblasComplex *)a, lda,
                               (const hipblasComplex *)b, ldb, (const hipblasComplex *)beta, (hipblasComplex *)c, ldc));

    hip_catch(hipDeviceSynchronize());
}

template <>
void gemm<hipDoubleComplex>(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipDoubleComplex *alpha,
                            const hipDoubleComplex *a, int lda, const hipDoubleComplex *b, int ldb, const hipDoubleComplex *beta,
                            hipDoubleComplex *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgemm(handle, transa, transb, m, n, k, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)a, lda,
                               (const hipblasDoubleComplex *)b, ldb, (const hipblasDoubleComplex *)beta, (hipblasDoubleComplex *)c, ldc));

    hip_catch(hipDeviceSynchronize());
}

template <>
void ger<float>(int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));

    hip_catch(hipDeviceSynchronize());
}

template <>
void ger<double>(int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda));

    hip_catch(hipDeviceSynchronize());
}

template <>
void ger<hipComplex>(int m, int n, const hipComplex *alpha, const hipComplex *x, int incx, const hipComplex *y, int incy, hipComplex *A,
                     int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgerc(handle, m, n, (const hipblasComplex *)alpha, (const hipblasComplex *)x, incx, (const hipblasComplex *)y,
                               incy, (hipblasComplex *)A, lda));

    hip_catch(hipDeviceSynchronize());
}

template <>
void ger<hipDoubleComplex>(int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *x, int incx, const hipDoubleComplex *y,
                           int incy, hipDoubleComplex *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgerc(handle, m, n, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)x, incx,
                               (const hipblasDoubleComplex *)y, incy, (hipblasDoubleComplex *)A, lda));

    hip_catch(hipDeviceSynchronize());
}

template <>
void gemv<float>(hipblasOperation_t transa, int m, int n, const float *alpha, const float *a, int lda, const float *x, int incx,
                 const float *beta, float *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));

    hip_catch(hipDeviceSynchronize());
}

template <>
void gemv<double>(hipblasOperation_t transa, int m, int n, const double *alpha, const double *a, int lda, const double *x, int incx,
                  const double *beta, double *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));

    hip_catch(hipDeviceSynchronize());
}

template <>
void gemv<hipComplex>(hipblasOperation_t transa, int m, int n, const hipComplex *alpha, const hipComplex *a, int lda, const hipComplex *x,
                      int incx, const hipComplex *beta, hipComplex *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgemv(handle, transa, m, n, (const hipblasComplex *)alpha, (const hipblasComplex *)a, lda,
                               (const hipblasComplex *)x, incx, (const hipblasComplex *)beta, (hipblasComplex *)y, incy));

    hip_catch(hipDeviceSynchronize());
}

template <>
void gemv<hipDoubleComplex>(hipblasOperation_t transa, int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
                            const hipDoubleComplex *x, int incx, const hipDoubleComplex *beta, hipDoubleComplex *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgemv(handle, transa, m, n, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)a, lda,
                               (const hipblasDoubleComplex *)x, incx, (const hipblasDoubleComplex *)beta, (hipblasDoubleComplex *)y, incy));

    hip_catch(hipDeviceSynchronize());
}

template <>
void scal<float>(int size, const float *alpha, float *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSscal(handle, size, alpha, x, incx));

    hip_catch(hipDeviceSynchronize());
}

template <>
void scal<double>(int size, const double *alpha, double *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDscal(handle, size, alpha, x, incx));

    hip_catch(hipDeviceSynchronize());
}

template <>
void scal<hipComplex>(int size, const hipComplex *alpha, hipComplex *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCscal(handle, size, (const hipblasComplex *)alpha, (hipblasComplex *)x, incx));

    hip_catch(hipDeviceSynchronize());
}

template <>
void scal<hipDoubleComplex>(int size, const hipDoubleComplex *alpha, hipDoubleComplex *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZscal(handle, size, (const hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)x, incx));

    hip_catch(hipDeviceSynchronize());
}

} // namespace detail

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T beta, CType<T, Rank> *C) {
    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    int m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    if constexpr (TransA && TransB && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        detail::gemm(HIPBLAS_OP_T, HIPBLAS_OP_T, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (TransA && TransB) {
        detail::gemm(HIPBLAS_OP_C, HIPBLAS_OP_C, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (TransA && !TransB && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        detail::gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (TransA && !TransB) {
        detail::gemm(HIPBLAS_OP_C, HIPBLAS_OP_N, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (!TransA && TransB && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        detail::gemm(HIPBLAS_OP_N, HIPBLAS_OP_T, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (!TransA && TransB) {
        detail::gemm(HIPBLAS_OP_N, HIPBLAS_OP_C, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else {
        detail::gemm(HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    }

    hip_catch(hipFree(alpha_gpu));
    hip_catch(hipFree(beta_gpu));
}

template <template <typename, size_t> typename CType, typename CDataType, template <typename, size_t> typename AType, typename ADataType,
          size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
void dot(CDataType C_prefactor, CType<CDataType, 0> &C,
         std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B, dim3 threads, dim3 blocks) {

    if (C_prefactor == CDataType{0}) {
        C.zero();
    } else {
        CDataType c;
        hip_catch(hipMemcpy(&c, C.data(), sizeof(CDataType), hipMemcpyDeviceToHost));

        c *= C_prefactor;
        hip_catch(hipMemcpy(C.data(), &c, sizeof(CDataType), hipMemcpyHostToDevice));
    }

    detail::dot_kernel<<<threads, blocks>>>(C.data(), AB_prefactor, A.data, B.data, A.size());

    hip_catch(hipDeviceSynchronize());
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires DeviceRankTensor<AType<T, ARank>, 2, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));

    detail::ger(X.dim(0), Y.dim(0), alpha_gpu, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));

    hip_catch(hipFree(alpha_gpu));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires DeviceRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y) {
    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    int m = A.dims(0), n = A.dims(1);

    if constexpr (TransA && (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        detail::gemv(HIPBLAS_OP_T, m, n, alpha_gpu, A.data(), A.strides(0), x.data(), x.strides(0), beta_gpu, y->data(), y->strides(0));
    } else if constexpr (TransA) {
        detail::gemv(HIPBLAS_OP_C, m, n, alpha_gpu, A.data(), A.strides(0), x.data(), x.strides(0), beta_gpu, y->data(), y->strides(0));
    } else {
        detail::gemv(HIPBLAS_OP_N, m, n, alpha_gpu, A.data(), A.strides(0), x.data(), x.strides(0), beta_gpu, y->data(), y->strides(0));
    }

    hip_catch(hipFree(alpha_gpu));
    hip_catch(hipFree(beta_gpu));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A) {
    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *scale_gpu;

    hip_catch(hipMalloc((void **)&scale_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(scale_gpu, &scale, sizeof(dev_datatype), hipMemcpyHostToDevice));

    detail::scal(A->size(), scale_gpu, A->data(), 1);

    hip_catch(hipFree(scale_gpu));
}

} // namespace linear_algebra
} // namespace gpu
} // namespace einsums