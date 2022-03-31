#pragma once

namespace einsums::backend::netlib {
int xerbla(const char *srname, int *info);
long int lsame(const char *ca, const char *cb);

int dgemm(char *, char *, int *, int *, int *, double *, const double *, int *, const double *, int *, double *, double *, int *);
} // namespace einsums::backend::netlib