#pragma once

#include "einsums/_GPUCast.hpp"
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
        if constexpr (std::is_same_v<CDataType, hipComplex> || std::is_same_v<CDataType, hipDoubleComplex>) {
            CDataType term = HipCast<CDataType, decltype(AB_prefactor)>::cast(AB_prefactor * A[curr_index] * B[curr_index]);
            atomicAdd(&(C->x), term.x);
            atomicAdd(&(C->y), term.y);
        } else {
            atomicAdd(C, HipCast<CDataType, decltype(AB_prefactor)>::cast(AB_prefactor * A[curr_index] * B[curr_index]));
        }
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
void gemm(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    int m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    hip_catch(hipMallocFromPoolAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_scale_pool(), stream));
    hip_catch(hipMallocFromPoolAsync((void **)&beta_gpu, sizeof(dev_datatype), get_scale_pool(), stream));

    hip_catch(hipMemcpyAsync((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, stream));
    hip_catch(hipMemcpyAsync((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice, stream));

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    hipStream_t prev_stream;
    hipblas_catch(hipblasGetStream(get_blas_handle(), &prev_stream));
    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));

    // Flip the A and B matrices. Row-major vs column major.
    detail::gemm(TransB ? HIPBLAS_OP_T : HIPBLAS_OP_N, TransA ? HIPBLAS_OP_T : HIPBLAS_OP_N, n, m, k, alpha_gpu, B.data(), ldb, A.data(),
                 lda, beta_gpu, C->data(), ldc);

    hip_catch(hipFreeAsync(alpha_gpu, stream));
    hip_catch(hipFreeAsync(beta_gpu, stream));

    hipblas_catch(hipblasSetStream(get_blas_handle(), prev_stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <template <typename, size_t> typename CType, typename CDataType, template <typename, size_t> typename AType, typename ADataType,
          size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
void dot(CDataType C_prefactor, CType<CDataType, 0> &C,
         ::std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B, hipStream_t stream) {
    using namespace einsums::gpu;

    if (C_prefactor == CDataType{0}) {
        C = CDataType{0};
    } else {
        C *= C_prefactor;
    }

    using dev_datatype = std::conditional_t<
        std::is_same_v<decltype(AB_prefactor), std::complex<float>>, hipComplex,
        std::conditional_t<std::is_same_v<decltype(AB_prefactor), std::complex<double>>, hipDoubleComplex, decltype(AB_prefactor)>>;

    detail::dot_kernel<<<block_size(A.size()), blocks(A.size()), 0, stream>>>(C.data(), HipCast<dev_datatype, decltype(AB_prefactor)>::cast(AB_prefactor), A.data(), B.data(),
                                            A.size());
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocFromPoolAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_scale_pool(), stream));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, stream));

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    hipStream_t prev_stream;
    hipblas_catch(hipblasGetStream(get_blas_handle(), &prev_stream));
    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));

    detail::ger(X.dim(0), Y.dim(0), alpha_gpu, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));

    hip_catch(hipFreeAsync(alpha_gpu, stream));

    hipblas_catch(hipblasSetStream(get_blas_handle(), prev_stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMallocFromPoolAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_scale_pool(), stream));
    hip_catch(hipMallocFromPoolAsync((void **)&beta_gpu, sizeof(dev_datatype), get_scale_pool(), stream));

    hip_catch(hipMemcpyAsync((void *)alpha_gpu, (const void *)&alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpyAsync((void *)beta_gpu, (const void *)&beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    int m = A.dim(1), n = A.dim(0);

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    hipStream_t prev_stream;
    hipblas_catch(hipblasGetStream(get_blas_handle(), &prev_stream));
    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));

    if constexpr (!TransA) {
        detail::gemv(HIPBLAS_OP_T, m, n, alpha_gpu, A.data(), A.stride(0), x.data(), x.stride(0), beta_gpu, y->data(), y->stride(0));
    } else {
        detail::gemv(HIPBLAS_OP_N, m, n, alpha_gpu, A.data(), A.stride(0), x.data(), x.stride(0), beta_gpu, y->data(), y->stride(0));
    }

    hip_catch(hipFreeAsync(alpha_gpu, stream));
    hip_catch(hipFreeAsync(beta_gpu, stream));

    hipblas_catch(hipblasSetStream(get_blas_handle(), prev_stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *scale_gpu;

    hip_catch(hipMallocFromPoolAsync((void **)&scale_gpu, sizeof(dev_datatype), get_scale_pool(), stream));

    hip_catch(hipMemcpyAsync(scale_gpu, &scale, sizeof(dev_datatype), hipMemcpyHostToDevice, stream));

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    hipStream_t prev_stream;
    hipblas_catch(hipblasGetStream(get_blas_handle(), &prev_stream));
    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));

    detail::scal(A->size(), scale_gpu, A->data(), 1);

    hip_catch(hipFreeAsync(scale_gpu, stream));
    
    hipblas_catch(hipblasSetStream(get_blas_handle(), prev_stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void symm_gemm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C, hipStream_t stream) {
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

    C->zero(stream);

    einsums::linear_algebra::gpu::detail::
        symm_gemm<<<block_size(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)), blocks(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)), 0, stream>>>(
            TransA, TransB, A.dim(0), C->dim(0), A.data(), A.stride(0), B.data(), B.stride(0), C->data(), C->stride(0));

}

} // namespace gpu
} // namespace linear_algebra
} // namespace einsums