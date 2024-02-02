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
namespace linear_algebra {
namespace gpu {

namespace detail {

template <typename CDataType, typename ADataType, typename BDataType>
__global__ void dot_kernel(CDataType *C, ::std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor,
                           const ADataType *A, const BDataType *B, size_t elements) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t curr_index = thread_id;

    while (curr_index < elements) {
        atomicAdd(C, (CDataType)(AB_prefactor * A[curr_index] * B[curr_index]));
        curr_index += kernel_size;
    }
}

} // namespace detail

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void gemm(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    using namespace einsums::gpu;
    
    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                            ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    int m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    if constexpr (TransA && TransB && (::std::is_same_v<T, float> || ::std::is_same_v<T, double>)) {
        detail::gemm(HIPBLAS_OP_T, HIPBLAS_OP_T, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (TransA && TransB) {
        detail::gemm(HIPBLAS_OP_C, HIPBLAS_OP_C, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (TransA && !TransB && (::std::is_same_v<T, float> || ::std::is_same_v<T, double>)) {
        detail::gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (TransA && !TransB) {
        detail::gemm(HIPBLAS_OP_C, HIPBLAS_OP_N, m, n, k, alpha_gpu, A.data(), lda, B.data(), ldb, beta_gpu, C->data(), ldc);
    } else if constexpr (!TransA && TransB && (::std::is_same_v<T, float> || ::std::is_same_v<T, double>)) {
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
         ::std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B, dim3 threads, dim3 blocks) {
            using namespace einsums::gpu;

    if (C_prefactor == CDataType{0}) {
        C = CDataType{0};
    } else {
        CDataType c;
        hip_catch(hipMemcpy(&c, C.data(), sizeof(CDataType), hipMemcpyDeviceToHost));

        c *= C_prefactor;
        hip_catch(hipMemcpy(C.data(), &c, sizeof(CDataType), hipMemcpyHostToDevice));
    }

    detail::dot_kernel<<<threads, blocks>>>(C.data(), AB_prefactor, A.data(), B.data(), A.size());

    hip_catch(hipDeviceSynchronize());
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    using namespace einsums::gpu;
    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                            ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));

    detail::ger(X.dim(0), Y.dim(0), alpha_gpu, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));

    hip_catch(hipFree(alpha_gpu));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y) {
    using namespace einsums::gpu;
    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                            ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    int m = A.dims(0), n = A.dims(1);

    if constexpr (TransA && (::std::is_same_v<T, float> || ::std::is_same_v<T, double>)) {
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
    requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A) {
    using namespace einsums::gpu;
    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                            ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *scale_gpu;

    hip_catch(hipMalloc((void **)&scale_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(scale_gpu, &scale, sizeof(dev_datatype), hipMemcpyHostToDevice));

    detail::scal(A->size(), scale_gpu, A->data(), 1);

    hip_catch(hipFree(scale_gpu));
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void gemmm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C) {
    using namespace einsums::gpu;
    assert(A.dim(0) == A.dim(1) && B.dim(0) == B.dim(1) && C->dim(0) == C->dim(1) && A.dim(0) == B.dim(0) && A.dim(0) == C->dim(0));

    C->zero();

    einsums::linear_algebra::gpu::detail::symm_gemm<<<dim3(64), dim3(64)>>>(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0), C->data(), C->stride(0));

    hip_catch(hipDeviceSynchronize());
}

} // namespace linear_algebra
} // namespace gpu
} // namespace einsums