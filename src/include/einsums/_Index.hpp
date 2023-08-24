#pragma once

#include "_Export.hpp"

#include <ostream>

#if defined(I)
#    undef I
#endif

namespace einsums::tensor_algebra::index {
struct LabelBase {};
} // namespace einsums::tensor_algebra::index

#define MAKE_INDEX(x)                                                                                                                      \
    namespace einsums::tensor_algebra::index {                                                                                             \
    struct x : public LabelBase {                                                                                                          \
        static constexpr char letter = static_cast<const char (&)[2]>(#x)[0];                                                              \
        constexpr x()                = default;                                                                                            \
    };                                                                                                                                     \
    static struct x x;                                                                                                                     \
    inline auto     operator<<(std::ostream &os, const struct x &) -> std::ostream     &{                                                     \
            os << x::letter;                                                                                                               \
            return os;                                                                                                                     \
    }                                                                                                                                      \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    struct EINSUMS_EXPORT fmt::formatter<struct ::einsums::tensor_algebra::index::x> : fmt::formatter<char> {                              \
        template <typename FormatContext>                                                                                                  \
        auto format(const struct ::einsums::tensor_algebra::index::x &, FormatContext &ctx) {                                              \
            return formatter<char>::format(::einsums::tensor_algebra::index::x::letter, ctx);                                              \
        }                                                                                                                                  \
    };

MAKE_INDEX(A); // NOLINT
MAKE_INDEX(a); // NOLINT
MAKE_INDEX(B); // NOLINT
MAKE_INDEX(b); // NOLINT
MAKE_INDEX(C); // NOLINT
MAKE_INDEX(c); // NOLINT
MAKE_INDEX(D); // NOLINT
MAKE_INDEX(d); // NOLINT
MAKE_INDEX(E); // NOLINT
MAKE_INDEX(e); // NOLINT
MAKE_INDEX(F); // NOLINT
MAKE_INDEX(f); // NOLINT
MAKE_INDEX(G); // NOLINT
MAKE_INDEX(g); // NOLINT
MAKE_INDEX(H); // NOLINT
MAKE_INDEX(h); // NOLINT
MAKE_INDEX(I); // NOLINT
MAKE_INDEX(i); // NOLINT
MAKE_INDEX(J); // NOLINT
MAKE_INDEX(j); // NOLINT
MAKE_INDEX(K); // NOLINT
MAKE_INDEX(k); // NOLINT
MAKE_INDEX(L); // NOLINT
MAKE_INDEX(l); // NOLINT
MAKE_INDEX(M); // NOLINT
MAKE_INDEX(m); // NOLINT
MAKE_INDEX(N); // NOLINT
MAKE_INDEX(n); // NOLINT
MAKE_INDEX(O); // NOLINT
MAKE_INDEX(o); // NOLINT
MAKE_INDEX(P); // NOLINT
MAKE_INDEX(p); // NOLINT
MAKE_INDEX(Q); // NOLINT
MAKE_INDEX(q); // NOLINT
MAKE_INDEX(R); // NOLINT
MAKE_INDEX(r); // NOLINT
MAKE_INDEX(S); // NOLINT
MAKE_INDEX(s); // NOLINT
MAKE_INDEX(T); // NOLINT
MAKE_INDEX(t); // NOLINT
MAKE_INDEX(U); // NOLINT
MAKE_INDEX(u); // NOLINT
MAKE_INDEX(V); // NOLINT
MAKE_INDEX(v); // NOLINT
MAKE_INDEX(w); // NOLINT
MAKE_INDEX(W); // NOLINT
MAKE_INDEX(x); // NOLINT
MAKE_INDEX(X); // NOLINT
MAKE_INDEX(y); // NOLINT
MAKE_INDEX(Y); // NOLINT
MAKE_INDEX(z); // NOLINT

// Z is a special index used internally. DO NOT USE.
MAKE_INDEX(Z); // NOLINT

#undef MAKE_INDEX

namespace einsums::tensor_algebra {

namespace index {

constexpr auto list = std::make_tuple(i, j, k, l, m, n, a, b, c, d, e, f, p, q, r, s);

} // namespace index

template <typename... Args>
struct EINSUMS_EXPORT Indices : public std::tuple<Args...> {
    Indices(Args... args) : std::tuple<Args...>(args...){};
};

} // namespace einsums::tensor_algebra