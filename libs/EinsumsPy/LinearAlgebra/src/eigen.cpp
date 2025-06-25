#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Utilities/InCollection.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void syev(std::string const &jobz, py::buffer &A, py::buffer &W) {
    py::buffer_info A_info = A.request(true), W_info = W.request(true);

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "A call to syev/heev can only take a rank-2 tensor as input!");
    }

    if (W_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "A call to syev/heev can only take a rank-1 tensor as output!");
    }

    char jobz_ch = 'N';

    if (jobz.length() >= 1) {
        if (is_in(jobz[0], {'n', 'N', 'v', 'V'})) {
            jobz_ch = toupper(jobz[0]);
        }
    }

    blas::int_t n     = A_info.shape[0];
    blas::int_t lda   = A_info.strides[0] / A_info.itemsize;
    blas::int_t lwork = 3 * n;

    // Type check
    if (A_info.format == py::format_descriptor<float>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<float>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != A_info.format) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<double>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != A_info.format) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<float>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != py::format_descriptor<float>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        if (W_info.format == py::format_descriptor<std::complex<double>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalues output from syev/heev are real, but the tensor passed is complex.");
        } else if (W_info.format != py::format_descriptor<double>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error,
                                    "The tensors passed to syev/heev need to have compatible storage types. Got A ({}), W ({}).",
                                    A_info.format, W_info.format);
        }
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The tensors passed to syev/heev have the wrong storage types. Got A ({}), W ({}).",
                                A_info.format, W_info.format);
    }

    blas::int_t info;

    // Calculate the size of the work array.
    if (A_info.format == py::format_descriptor<float>::format()) {
        float lwork_temp;
        info  = blas::syev<float>(jobz_ch, 'U', n, (float *)A_info.ptr, lda, (float *)W_info.ptr, &lwork_temp, -1);
        lwork = lwork_temp;
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        double lwork_temp;
        info  = blas::syev<double>(jobz_ch, 'U', n, (double *)A_info.ptr, lda, (double *)W_info.ptr, &lwork_temp, -1);
        lwork = lwork_temp;
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        std::complex<float> lwork_temp;
        info = blas::heev<float>(jobz_ch, 'U', n, (std::complex<float> *)A_info.ptr, lda, (float *)W_info.ptr, &lwork_temp, (blas::int_t)-1,
                                 (float *)nullptr);
        lwork = lwork_temp.real();
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        std::complex<double> lwork_temp;
        info  = blas::heev<double>(jobz_ch, 'U', n, (std::complex<double> *)A_info.ptr, lda, (double *)W_info.ptr, &lwork_temp,
                                   (blas::int_t)-1, (double *)nullptr);
        lwork = lwork_temp.real();
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "Can only perform gemv on floating point matrices! Got type {}.", A_info.format);
    }

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument had an illegal value!", print::ordinal<blas::int_t>(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
    }

    std::vector<char, BufferAllocator<char>> work_vec(lwork * A_info.itemsize);
    void                                    *work = (void *)work_vec.data();

    if (work == nullptr) {
        EINSUMS_THROW_EXCEPTION(
            std::runtime_error,
            "Could not allocate work array for syev call! Error unknown, but likely due to lack of memory for allocation.");
    }

    if (A_info.format == py::format_descriptor<float>::format()) {
        info = blas::syev<float>(jobz_ch, 'U', n, (float *)A_info.ptr, lda, (float *)W_info.ptr, (float *)work, lwork);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        info = blas::syev<double>(jobz_ch, 'U', n, (double *)A_info.ptr, lda, (double *)W_info.ptr, (double *)work, lwork);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        std::vector<float, BufferAllocator<float>> rwork_vec(std::max(3 * n - 2, blas::int_t{1}));

        info = blas::heev<float>(jobz_ch, 'U', n, (std::complex<float> *)A_info.ptr, lda, (float *)W_info.ptr, (std::complex<float> *)work,
                                 lwork, rwork_vec.data());
        // Fix the eigenvectors. They should be conjugated due to the difference between row and column major.
        if (info == 0) {
            std::complex<float> *A_data = (std::complex<float> *)A_info.ptr;
#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    pragma omp parallel for simd collapse(2)
#else
#    pragma omp parallel for collapse(2)
#endif
            for (size_t i = 0; i < n * lda; i += lda) {
                for (size_t j = 0; j < n; j++) {
                    A_data[i + j] = std::conj(A_data[i + j]);
                }
            }
        }
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        std::vector<double, BufferAllocator<double>> rwork_vec(std::max(3 * n - 2, blas::int_t{1}));

        info = blas::heev<double>(jobz_ch, 'U', n, (std::complex<double> *)A_info.ptr, lda, (double *)W_info.ptr,
                                  (std::complex<double> *)work, lwork, rwork_vec.data());

        // Fix the eigenvectors. They should be conjugated due to the difference between row and column major.
        if (info == 0) {
            std::complex<double> *A_data = (std::complex<double> *)A_info.ptr;
#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    pragma omp parallel for simd collapse(2)
#else
#    pragma omp parallel for collapse(2)
#endif
            for (size_t i = 0; i < n * lda; i += lda) {
                for (size_t j = 0; j < n; j++) {
                    A_data[i + j] = std::conj(A_data[i + j]);
                }
            }
        }
    }

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument had an illegal value!", print::ordinal<blas::int_t>(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
    }
}

template <typename T>
void geev_real_in_cmplx_out(char jobvl, char jobvr, py::buffer &A, py::buffer &W, py::buffer_info &Vl_info, py::buffer_info &Vr_info) {
    py::buffer_info A_info = A.request(true), W_info = W.request(true);

    blas::int_t n = A_info.shape[0], lda = A_info.strides[0] / A_info.itemsize;
    blas::int_t ldvl = n, ldvr = n;
    blas::int_t lwork = 3 * n;

    // All checks done beforehand. Create buffers.
    std::vector<T, BufferAllocator<T>> Vl_real, Vr_real;
    T                                 *Vl_real_ptr = nullptr, *Vr_real_ptr = nullptr;

    std::complex<T> *W_data = reinterpret_cast<std::complex<T> *>(W_info.ptr);

    if (jobvl == 'V') {
        Vl_real.resize(n * n);
        Vl_real_ptr = Vl_real.data();
    }

    if (jobvr == 'V') {
        Vr_real.resize(n * n);
        Vr_real_ptr = Vr_real.data();
    }

    blas::int_t info = 0;

    EINSUMS_LOG_DEBUG("geev parameters: jobvl = {}, jobvr = {}, n = {}, lda = {}, ldvl = {}, ldvr = {}.", jobvl, jobvr, n, lda, ldvl, ldvr);

    // Decompose.
    info = blas::geev<T>(jobvl, jobvr, n, (T *)A_info.ptr, lda, W_data, Vl_real_ptr, ldvl, Vr_real_ptr, ldvr);

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument had an illegal value!", print::ordinal<blas::int_t>(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
    }

    // Now, the left eigenvectors.
    if (jobvl == 'V') {
        std::complex<T> *Vl_data = reinterpret_cast<std::complex<T> *>(Vl_info.ptr);
        for (size_t j = 0; j < n; j++) {
            // Handle the complex case.
            if (W_data[j].imag() != T{0.0} && j < n - 1) {
                for (size_t i = 0; i < n; i++) {
                    // These are complex conjugate pairs.
                    Vl_data[j * ldvl + i]       = std::complex<T>(Vl_real_ptr[i * ldvl + j], Vl_real_ptr[i * ldvl + j + 1]);
                    Vl_data[(j + 1) * ldvl + i] = std::complex<T>(Vl_real_ptr[i * ldvl + j], -Vl_real_ptr[i * ldvl + j + 1]);
                }
                // Skip the next case.
                j++;
            } else {
                for (size_t i = 0; i < n; i++) {
                    Vl_data[j * ldvl + i] = std::complex<T>(Vl_real_ptr[j * ldvl + i], 0);
                }
            }
        }
    }

    // Now, the right eigenvectors.
    if (jobvr == 'V') {
        std::complex<T> *Vr_data = reinterpret_cast<std::complex<T> *>(Vr_info.ptr);
        for (size_t j = 0; j < n; j++) {
            // Handle the complex case.
            if (W_data[j].imag() != T{0.0} && j < n - 1) {
                for (size_t i = 0; i < n; i++) {
                    // These are complex conjugate pairs.
                    Vr_data[i * ldvl + j]     = std::complex<T>(Vr_real_ptr[i * ldvl + j], Vr_real_ptr[i * ldvl + j + 1]);
                    Vr_data[i * ldvl + j + 1] = std::complex<T>(Vr_real_ptr[i * ldvl + j], -Vr_real_ptr[i * ldvl + j + 1]);
                }
                // Skip the next case.
                j++;
            } else {
                for (size_t i = 0; i < n; i++) {
                    Vr_data[i * ldvl + j] = std::complex<T>(Vr_real_ptr[i * ldvl + j], 0);
                }
            }
        }
    }
}

template <typename T>
void geev_cmplx_in_cmplx_out(char jobvl, char jobvr, py::buffer &A, py::buffer &W, py::buffer_info &Vl_info, py::buffer_info &Vr_info) {
    py::buffer_info A_info = A.request(true), W_info = W.request(true);

    blas::int_t n = A_info.shape[0], lda = A_info.strides[0] / A_info.itemsize;
    blas::int_t ldvl = n, ldvr = n;
    blas::int_t lwork = 2 * n;

    blas::int_t info = 0;

    std::vector<RemoveComplexT<T>, BufferAllocator<RemoveComplexT<T>>> rwork(2 * n);

    T *Vl_ptr = nullptr, *Vr_ptr = nullptr;

    if (jobvl == 'V') {
        Vl_ptr = reinterpret_cast<T *>(Vl_info.ptr);
    }

    if (jobvr == 'V') {
        Vr_ptr = reinterpret_cast<T *>(Vr_info.ptr);
    }

    // Decompose.
    info = blas::geev<T>(jobvl, jobvr, n, (T *)A_info.ptr, lda, (T *)W_info.ptr, Vl_ptr, ldvl, Vr_ptr, ldvr);

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The {} argument had an illegal value!", print::ordinal<blas::int_t>(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
    }

    // Transpose the outputs.
    // if (jobvl == 'V') {
    //     for (size_t i = 0; i < n; i++) {
    //         for (size_t j = i + 1; j < n; j++) {
    //             std::swap(Vl_ptr[i * ldvl + j], Vl_ptr[j * ldvl + i]);
    //         }
    //     }
    // }

    // if (jobvr == 'V') {
    //     for (size_t i = 0; i < n; i++) {
    //         for (size_t j = i + 1; j < n; j++) {
    //             std::swap(Vr_ptr[i * ldvr + j], Vr_ptr[j * ldvr + i]);
    //         }
    //     }
    // }
}

void geev(std::string const &jobvl, std::string const &jobvr, py::buffer &A, py::buffer &W,
          std::variant<pybind11::buffer, pybind11::none> &Vl_or_none, std::variant<pybind11::buffer, pybind11::none> &Vr_or_none) {
    char jobvl_ch = 'n', jobvr_ch = 'n';

    if (jobvl.length() > 0 && is_in(jobvl, {"n", "N", "v", "V"})) {
        jobvl_ch = toupper(jobvl[0]);
    }

    if (jobvr.length() > 0 && is_in(jobvr, {"n", "N", "v", "V"})) {
        jobvr_ch = toupper(jobvr[0]);
    }

    py::buffer_info A_info = A.request(false), W_info = W.request(false), Vl_info, Vr_info;

    if (A_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only perform eigendecomposition on matrices!");
    }

    if (W_info.ndim != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The output of geev is a vector of eigenvalues!");
    }

    if (A_info.shape[0] != A_info.shape[1]) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can only perform eigendecomposition on square matrices!");
    }

    if (A_info.shape[0] != W_info.shape[0]) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                "The number of eigenvalues must be the same as the length along one dimension of the input matrix!");
    }

    if (jobvl_ch == 'V') {
        if (std::holds_alternative<py::none>(Vl_or_none)) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Left eigenvectors were requested, but None was passed as the argument!");
        }

        Vl_info = std::get<pybind11::buffer>(Vl_or_none).request(true);

        if (Vl_info.ndim != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "The left eigenvectors form a matrix!");
        }

        if (Vl_info.shape[0] != Vl_info.shape[1]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The left eigenvector matrix needs to be square!");
        }
        if (A_info.shape[0] != Vl_info.shape[0]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The left eigenvector matrix needs to have the same dimensions as the input matrix!");
        }
    } else {
        Vl_info.format   = "";
        Vl_info.itemsize = 0;
        Vl_info.ndim     = 0;
        Vl_info.ptr      = nullptr;
        Vl_info.readonly = true;
        Vl_info.shape.clear();
        Vl_info.size = 0;
        Vl_info.strides.clear();
    }

    if (jobvr_ch == 'V') {
        if (std::holds_alternative<py::none>(Vr_or_none)) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "Left eigenvectors were requested, but None was passed as the argument!");
        }

        Vr_info = std::get<pybind11::buffer>(Vr_or_none).request(true);

        if (Vr_info.ndim != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "The right eigenvectors form a matrix!");
        }

        if (Vr_info.shape[0] != Vr_info.shape[1]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The right eigenvector matrix needs to be square!");
        }
        if (A_info.shape[0] != Vr_info.shape[0]) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The right eigenvector matrix needs to have the same dimensions as the input matrix!");
        }
    } else {
        Vr_info.format   = "";
        Vr_info.itemsize = 0;
        Vr_info.ndim     = 0;
        Vr_info.ptr      = nullptr;
        Vr_info.readonly = true;
        Vr_info.shape.clear();
        Vr_info.size = 0;
        Vr_info.strides.clear();
    }

    if ((W_info.format != Vl_info.format && jobvl_ch == 'V') || (W_info.format != Vr_info.format && jobvr_ch == 'V')) {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The eigenvalue buffer and the eigenvector buffers that will be used need to have the same "
                                                "storage type! Unused buffers are ignored.");
    }

    if (A_info.format == py::format_descriptor<float>::format()) {
        if (W_info.format == py::format_descriptor<float>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The output buffers need to have complex data types to support the eigenvalues.");
        } else if (W_info.format != py::format_descriptor<std::complex<float>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The output buffers have an incompatible storage type!");
        } else {
            geev_real_in_cmplx_out<float>(jobvl_ch, jobvr_ch, A, W, Vl_info, Vr_info);
        }
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        if (W_info.format == py::format_descriptor<double>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The output buffers need to have complex data types to support the eigenvalues.");
        } else if (W_info.format != py::format_descriptor<std::complex<double>>::format()) {
            EINSUMS_THROW_EXCEPTION(py::type_error, "The output buffers have an incompatible storage type!");
        } else {
            geev_real_in_cmplx_out<double>(jobvl_ch, jobvr_ch, A, W, Vl_info, Vr_info);
        }
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format() &&
               W_info.format == py::format_descriptor<std::complex<float>>::format()) {
        geev_cmplx_in_cmplx_out<std::complex<float>>(jobvl_ch, jobvr_ch, A, W, Vl_info, Vr_info);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format() &&
               W_info.format == py::format_descriptor<std::complex<double>>::format()) {
        geev_cmplx_in_cmplx_out<std::complex<double>>(jobvl_ch, jobvr_ch, A, W, Vl_info, Vr_info);
    } else {
        EINSUMS_THROW_EXCEPTION(py::type_error, "The input and output buffers have incompatible storage types!");
    }
}

} // namespace detail
} // namespace python
} // namespace einsums