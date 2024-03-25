

#include "einsums/_Common.hpp"

#include <numeric>
BEGIN_EINSUMS_NAMESPACE_CPP(einsums::symm_index)

namespace detail {

EINSUMS_EXPORT size_t calc_index(size_t n, size_t k) {
    if (k == 1) {
        return n;
    } else if (k == 2) {
        return (n * (n + 1)) / 2;
    } else {
        size_t num = 1, den = 1;

        for (int i = 0; i < k; i++) {
            num *= n + i;
            den *= i + 1;

            size_t div = std::gcd(num, den);

            num /= div;
            den /= div;
        }

        return num;
    }
}

} // namespace detail

END_EINSUMS_NAMESPACE_CPP(einsums::symm_index)
