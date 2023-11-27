#pragma once

#include "einsums/_Common.hpp"

#include "einsums/ElementOperations.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/utility/TensorTraits.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::polynomial::laguerre)

template <typename T>
auto companion(const Tensor<T, 1> &c) -> std::enable_if_t<std::is_signed_v<T>, Tensor<T, 2>> {
    LabeledSection0();

    if (c.dim(0) < 2) {
        throw std::runtime_error("Series (c) must have maximum degree of at least 1.");
    }

    if (c.dim(0) == 2) {
        auto result  = create_tensor<T>("Laguerre companion matrix", 1, 1);
        result(0, 0) = T{1} + (c(0) / c(1));
        return result;
    }

    auto n   = c.dim(0) - 1;
    auto mat = create_tensor<T>("Laguerre companion matrix", n, n);
    zero(mat);

    auto mat1 = TensorView{mat, Dim<1>{-1}};
    auto top  = TensorView{mat1, Dim<1>{-1}, Offset<1>{1}, Stride<1>{n + 1}};
    auto mid  = TensorView{mat1, Dim<1>{-1}, Offset<1>{0}, Stride<1>{n + 1}};
    auto bot  = TensorView{mat1, Dim<1>{-1}, Offset<1>{n}, Stride<1>{n + 1}};

    auto U = arange<T>(1, n);
    U *= T{-1.0};
    top = U;

    auto V = arange<T>(n);
    V *= T{2.0};
    V += T{1.0};
    mid = V;

    bot = U;

    // TODO: Need to add c's contribution to mat. In python:
    // mat[:, -1] += (c[:-1]/c[-1])*n

    return mat;
}

template <typename T>
auto derivative(const Tensor<T, 1> &_c, unsigned int m = 1, T scale = T{1}) -> Tensor<T, 1> {
    LabeledSection0();

    Tensor<T, 1> c = _c;
    c.set_name("c derivative");

    if (m == 0)
        return c;

    auto n = c.dim(0);
    if (m >= n) {
        Tensor<T, 1> result{_c.name(), c.dim(0) - 1};
        zero(result);
        return result;
    }

    for (int i = 0; i < m; i++) {
        n -= 1;
        c *= scale;
        Tensor<T, 1> der{_c.name(), n};
        zero(der);
        for (int j = n; j >= 1; j -= 1) {
            der(j - 1) = -c(j);
            c(j - 1) += c(j);
        }
        der(0) = -c(1);
        c      = der;
    }

    return c;
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename CType, typename T>
    requires requires {
        requires CoreRankTensor<XType<T, 1>, 1, T>;
        requires CoreRankTensor<CType<T, 1>, 1, T>;
    }
auto value(const XType<T, 1> &x, const CType<T, 1> &c) -> Tensor<T, 1> {
    LabeledSection0();

    auto c0 = create_tensor_like("c0", x), c1 = create_tensor_like("c1", x);
    zero(c0);
    zero(c1);

    if (c.dim(0) == 1) {
        c0 = c(0);
        c1 = T{0};
    } else if (c.dim(0) == 2) {
        c0 = c(0);
        c1 = c(1);
    } else {
        size_t nd = c.dim(0);

        c0 = c(-2);
        c1 = c(-1);

        auto tmp  = create_tensor_like("tmp", x);
        auto tmp1 = create_tensor_like("tmp1", x);
        for (int i = 3; i < c.dim(0) + 1; i++) {
            tmp = c0;
            nd  = nd - 1;

            c0 = c1;
            c0 *= (nd - 1);
            c0 /= nd;
            c0 *= -1.0;
            c0 += c(-i);

            tmp1 = (2 * nd - 1);
            tmp1 -= x;
            tmp1 *= c1;
            tmp1 /= nd;
            c1 = tmp;
            c1 += tmp1;
        }
    }

    auto result = create_tensor_like("laguerre_value", x);
    result      = T{1};
    result -= x;
    result *= c1;
    result += c0;

    return result;
}

template <typename T = double>
auto gauss_laguerre(unsigned int degree) -> std::tuple<Tensor<T, 1>, Tensor<T, 1>> {
    LabeledSection0();

    // First approximation of roots. We use the fact that the companion matrix is symmetric in this case in order to obtain better zeros.
    auto c = create_tensor<double>("c", degree + 1);
    zero(c);
    c(-1)  = 1.0;
    auto m = companion(c);
    auto x = create_tensor<double>("x", degree);
    zero(x);

    linear_algebra::syev(&m, &x);

    // Improve roots by one application of Newtown.
    auto dy = value(x, c);
    auto df = value(x, derivative(c));

    dy /= df;
    x -= dy;

    auto fm = value(x, TensorView{c, Dim<1>{-1}, Offset<1>{1}});
    fm.set_name("fm");

    // Scale the factor to avoid possible numerical overflow
    using namespace einsums::element_operations::new_tensor;

    fm /= max(abs(fm));
    df /= max(abs(df));

    fm *= df;

    auto w = invert(fm);
    w /= sum(w);

    w.set_name("w");

    return std::make_tuple(x, w);
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto weight(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    LabeledSection0();

    auto result = create_tensor_like(tensor);
    result      = tensor;
    auto &data  = result.vector_data();

    element_operations::detail::omp_loop(data, [&](T &value) { return std::exp(-value); });

    return result;
}

END_EINSUMS_NAMESPACE_HPP(einsums::polynomial::laguerre)