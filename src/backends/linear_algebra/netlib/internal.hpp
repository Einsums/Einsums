#pragma once

namespace einsums::backend::linear_algebra::netlib {
auto xerbla(const char *srname, int *info) -> int;
auto lsame(const char *ca, const char *cb) -> long int;
} // namespace einsums::backend::linear_algebra::netlib