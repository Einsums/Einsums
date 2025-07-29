#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename T>
void impl_qr_tridiagonal(einsums::detail::TensorImpl<T> *A, T *diag, T *subdiag) {
    A->zero();
    
    A->subscript(0, 0) = T{1.0};

    
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums