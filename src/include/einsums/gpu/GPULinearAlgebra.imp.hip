#pragma once

#include "einsums/_GPUCast.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/../../backends/linear_algebra/hipblas/hipblas.hpp"
#include "einsums/DeviceTensor.hpp"
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
        requires !::einsums::detail::DeviceRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (::einsums::detail::DeviceRankBlockTensor<AType<T, Rank>, 2, T> &&
                      ::einsums::detail::DeviceRankBlockTensor<BType<T, Rank>, 2, T>);
    }
void gemm(const T *alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T *beta, CType<T, Rank> *C, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<CType<T, Rank>, Rank, T>) {

        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
            throw std::runtime_error("gemm: Tensors need the same number of blocks.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemm<TransA, TransB>(alpha, A.block(i), B.block(i), beta, &(C->block(i)), stream);
        }

        return;
    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                         einsums::detail::IsDeviceRankBlockTensorV<BType<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks()) {
            gemm<TransA, TransB>(alpha, (Tensor<T, 2>)A, (Tensor<T, 2>)B, beta, C, stream);
        } else {
            EINSUMS_OMP_PARALLEL_FOR
            for (int i = 0; i < A.num_blocks(); i++) {
                if (A.block_dim(i) == 0) {
                    continue;
                }
                gemm<TransA, TransB>(alpha, A.block(i), B.block(i), beta, &((*C)(A.block_range(i), A.block_range(i))), stream);
            }
        }

        return;
    } else {

        using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                                  ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

        int m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
        int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

        hipEvent_t wait_event;

        hip_catch(hipEventCreate(&wait_event));
        hip_catch(hipEventRecord(wait_event, stream));

        // Wait for the copies to happen, since the values will disappear later.
        hip_catch(hipEventSynchronize(wait_event));

        hipStream_t prev_stream;
        hipblas_catch(hipblasGetStream(get_blas_handle(), &prev_stream));
        hipblas_catch(hipblasSetStream(get_blas_handle(), stream));

        // Flip the A and B matrices. Row-major vs column major.
        detail::gemm(TransB ? HIPBLAS_OP_T : HIPBLAS_OP_N, TransA ? HIPBLAS_OP_T : HIPBLAS_OP_N, n, m, k, (dev_datatype *)alpha, B.data(),
                     ldb, A.data(), lda, (dev_datatype *)beta, C->data(), ldc);

        hipblas_catch(hipblasSetStream(get_blas_handle(), prev_stream));

        hip_catch(hipEventDestroy(wait_event));
    }
}

template <template <typename, size_t> typename CType, typename CDataType, template <typename, size_t> typename AType, typename ADataType,
          size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
void dot(CDataType C_prefactor, CType<CDataType, 0> &C,
         ::std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B, hipStream_t stream) {
    using namespace einsums::gpu;

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        if (A.num_blocks() != B.num_blocks()) {
            dot(C_prefactor, C, (einsums::DeviceTensor<ADataType, ARank>)A, (einsums::DeviceTensor<BDataType, BRank>)B, stream);
        }

        if (A.ranges() != B.ranges()) {
            dot(C_prefactor, C, (einsums::DeviceTensor<ADataType, ARank>)A, (einsums::DeviceTensor<BDataType, BRank>)B, stream);
        }

        if (C_prefactor == CDataType{0}) {
            C = CDataType{0};
        } else {
            C *= C_prefactor;
        }

#pragma omp parallel for reduction(+ : C)
        for (int i = 0; i < A.num_blocks(); i++) {

            if (A.block_dim(i) == 0) {
                continue;
            }
            CType<CDataType, 0> temp;
            dot(CDataType{0}, temp, A.block(i), B.block(i));

            C += temp;
        }

    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                         !einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        dot(C_prefactor, C, (einsums::DeviceTensor<ADataType, ARank>)A, B, stream);
    } else if constexpr (!einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                         einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        dot(C_prefactor, C, A, (einsums::DeviceTensor<BDataType, BRank>)B, stream);

    } else {

        if (C_prefactor == CDataType{0}) {
            C = CDataType{0};
        } else {
            C *= C_prefactor;
        }

        using dev_datatype = std::conditional_t<
            std::is_same_v<decltype(AB_prefactor), std::complex<float>>, hipComplex,
            std::conditional_t<std::is_same_v<decltype(AB_prefactor), std::complex<double>>, hipDoubleComplex, decltype(AB_prefactor)>>;

        detail::dot_kernel<<<block_size(A.size()), blocks(A.size()), 0, stream>>>(
            C.data(), HipCast<dev_datatype, decltype(AB_prefactor)>::cast(AB_prefactor), A.data(), B.data(), A.size());
    }
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires !::einsums::detail::DeviceRankBlockTensor<AType<T, ARank>, 2, T>;
    }
void ger(const T *alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    hipStream_t prev_stream;
    hipblas_catch(hipblasGetStream(get_blas_handle(), &prev_stream));
    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));

    detail::ger(X.dim(0), Y.dim(0), (dev_datatype *)alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));

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
void gemv(const T *alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, const T *beta, YType<T, XYRank> *y, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemv(alpha, A.block(i), x(A.block_range(i)), beta, &((*y)(A.block_range(i))));
        }

    } else {

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
            detail::gemv(HIPBLAS_OP_T, m, n, (dev_datatype *)alpha, A.data(), A.stride(0), x.data(), x.stride(0), (dev_datatype *)beta,
                         y->data(), y->stride(0));
        } else {
            detail::gemv(HIPBLAS_OP_N, m, n, (dev_datatype *)alpha, A.data(), A.stride(0), x.data(), x.stride(0), (dev_datatype *)beta,
                         y->data(), y->stride(0));
        }

        hipblas_catch(hipblasSetStream(get_blas_handle(), prev_stream));

        hip_catch(hipEventDestroy(wait_event));
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(const T *scale, AType<T, ARank> *A, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    hipStream_t prev_stream;
    hipblas_catch(hipblasGetStream(get_blas_handle(), &prev_stream));
    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));

    detail::scal(A->size(), (dev_datatype *)scale, A->data(), 1);

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

    einsums::linear_algebra::gpu::detail::symm_gemm<<<block_size(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)),
                                                      blocks(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)), 0, stream>>>(
        TransA, TransB, A.dim(0), C->dim(0), A.data(), A.stride(0), B.data(), B.stride(0), C->data(), C->stride(0));
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<CType<T, Rank>, 2, T>;
        requires !::einsums::detail::DeviceRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (::einsums::detail::DeviceRankBlockTensor<AType<T, Rank>, 2, T> &&
                      ::einsums::detail::DeviceRankBlockTensor<BType<T, Rank>, 2, T>);
        requires(!std::is_pointer_v<T>);
    }
void gemm(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMallocFromPoolAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_scale_pool(), stream));
    hip_catch(hipMallocFromPoolAsync((void **)&beta_gpu, sizeof(dev_datatype), get_scale_pool(), stream));

    hip_catch(hipMemcpyAsync((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, stream));
    hip_catch(hipMemcpyAsync((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice, stream));

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    // Flip the A and B matrices. Row-major vs column major.
    gemm<TransA, TransB>((T *)alpha_gpu, A, B, (T *)beta_gpu, C, stream);

    hip_catch(hipFreeAsync(alpha_gpu, stream));
    hip_catch(hipFreeAsync(beta_gpu, stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires !::einsums::detail::DeviceRankBlockTensor<AType<T, ARank>, 2, T>;
        requires !std::is_pointer_v<T>;
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

    ger((T *)alpha_gpu, X, Y, A, stream);

    hip_catch(hipFreeAsync(alpha_gpu, stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::detail::DeviceRankTensor<YType<T, XYRank>, 1, T>;
        requires !std::is_pointer_v<T>;
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

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    gemv<TransA>((T *)alpha_gpu, A, x, (T *)beta_gpu, y, stream);

    hip_catch(hipFreeAsync(alpha_gpu, stream));
    hip_catch(hipFreeAsync(beta_gpu, stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, ARank, T>;
        requires !std::is_pointer_v<T>;
    }
void scale(T scale_, AType<T, ARank> *A, hipStream_t stream) {
    using namespace einsums::gpu;
    using namespace einsums::backend::linear_algebra::hipblas::detail;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *scale_gpu;

    hip_catch(hipMallocFromPoolAsync((void **)&scale_gpu, sizeof(dev_datatype), get_scale_pool(), stream));

    hip_catch(hipMemcpyAsync(scale_gpu, &scale_, sizeof(dev_datatype), hipMemcpyHostToDevice, stream));

    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    hip_catch(hipEventSynchronize(wait_event));

    scale((T *)scale_gpu, A, stream);

    hip_catch(hipFreeAsync(scale_gpu, stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires requires {
        requires ::einsums::detail::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::detail::DeviceRankTensor<BType<T, BRank>, 2, T>;
    }
int gesv(AType<T, ARank> *A, BType<T, BRank> *B, hipStream_t stream) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<BType<T, BRank>, BRank, T>) {

        if (A->num_blocks() != B->num_blocks()) {
            throw std::runtime_error("gesv: Tensors need the same number of blocks.");
        }

        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &(B->block(i)), stream);

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;

    } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {

            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &((*B)(AllT(), A->block_range(i))), stream);

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;
    }

    LabeledSection0();

    auto n   = A->dim(0);
    auto lda = A->stride(0);
    auto ldb = B->stride(0);

    auto nrhs = B->dim(0);

    int lwork = n;

    __device_ptr__ int *ipiv;

#ifdef __HIP_PLATFORM_NVIDIA__
    DeviceTensor<T, BRank> X = DeviceTensor<T, BRank>(einsums::detail::DEV_ONLY, B->dims());
#endif

    einsums::gpu::hip_catch(hipMallocAsync((void **)&ipiv, sizeof(int) * lwork, stream));

    hipEvent_t wait_event;

    einsums::gpu::hip_catch(hipEventCreate(&wait_event));
    einsums::gpu::hip_catch(hipEventRecord(wait_event, stream));

    // Wait for the copies to happen, since the values will disappear later.
    einsums::gpu::hip_catch(hipEventSynchronize(wait_event));

#ifdef __HIP_PLATFORM_NVIDIA__
    int info = detail::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb, X.data(), X.stride(0));
#elif defined(__HIP_PLATFORM_AMD__)
    int info = detail::gesv(n, nrhs, A->data(), lda, ipiv, B->data(), ldb, B->data(), ldb);
#endif

    einsums::gpu::hip_catch(hipFreeAsync(ipiv, stream));

    einsums::gpu::hip_catch(hipEventDestroy(wait_event));

    return info;
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires(!std::is_pointer_v<T>)
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y, hipStream_t stream) {
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

    axpy((T *)alpha, X, Y, stream);

    hip_catch(hipFreeAsync(alpha, stream));

    hip_catch(hipEventDestroy(wait_event));
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
void axpy(const T *alpha, const XType<T, Rank> &X, YType<T, Rank> *Y, hipStream_t stream) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<XType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<YType<T, Rank>, Rank, T>) {
        if (X.num_blocks() != Y->num_blocks()) {
            throw std::runtime_error("axpy: Tensors need to have the same number of blocks.");
        }

        if (X.ranges() != Y->ranges()) {
            throw std::runtime_error("axpy: Tensor blocks need to be compatible.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim() == 0) {
                continue;
            }

            axpy(alpha, X[i], &(Y->block(i)));
        }
    } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<XType<T, Rank>, Rank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim() == 0) {
                continue;
            }

            std::array<einsums::Range, Rank> slice;

            slice.fill(X.block_range());

            auto Y_block = std::apply(*Y, slice);

            axpy(alpha, X[i], &Y_block);
        }
    } else {

        LabeledSection0();

        detail::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
    }
}

} // namespace gpu
} // namespace linear_algebra
} // namespace einsums