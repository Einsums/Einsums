#pragma once

namespace einsums::backend::netlib {
auto xerbla(const char *srname, int *info) -> int;
auto lsame(const char *ca, const char *cb) -> long int;
} // namespace einsums::backend::netlib