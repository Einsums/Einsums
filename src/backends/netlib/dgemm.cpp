/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
        -lf2c -lm   (in that order)
*/

#include "Netlib.hpp"
#include "einsums/Print.hpp"
#include "einsums/Timer.hpp"
#include "internal.hpp"

#include <algorithm>

namespace einsums::backend::netlib {

namespace {
/* Subroutine */
auto _dgemm(char *transa, char *transb, int *m, int *n, int *k, double *alpha, const double *a, int *lda, const double *b, int *ldb,
            double *beta, double *c, int *ldc) -> int {
    Timer::Timer timer("netlib::dgemm");
    println("dgemm: ta({}) tb({}) m({}) n({}) k({}) alpha({}), lda({}), ldb({}), beta({}), ldc({})", *transa, *transb, *m, *n, *k, *alpha,
            *lda, *ldb, *beta, *ldc);
    println("first elements C({}) A({}) B({})", c[0], a[0], b[0]);

    /* System generated locals */
    int a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, i__3;

    /* Local variables */
    int info;
    long int nota, notb;
    double temp;
    int i, j, l, ncola;
    int nrowa, nrowb;

    /*  Purpose
        =======

        DGEMM  performs one of the matrix-matrix operations

           C := alpha*op( A )*op( B ) + beta*C,

        where  op( X ) is one of

           op( X ) = X   or   op( X ) = X',

        alpha and beta are scalars, and A, B and C are matrices, with op( A )

        an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.


        Parameters
        ==========

        TRANSA - CHARACTER*1.
                 On entry, TRANSA specifies the form of op( A ) to be used in

                 the matrix multiplication as follows:

                    TRANSA = 'N' or 'n',  op( A ) = A.

                    TRANSA = 'T' or 't',  op( A ) = A'.

                    TRANSA = 'C' or 'c',  op( A ) = A'.

                 Unchanged on exit.

        TRANSB - CHARACTER*1.
                 On entry, TRANSB specifies the form of op( B ) to be used in

                 the matrix multiplication as follows:

                    TRANSB = 'N' or 'n',  op( B ) = B.

                    TRANSB = 'T' or 't',  op( B ) = B'.

                    TRANSB = 'C' or 'c',  op( B ) = B'.

                 Unchanged on exit.

        M      - INTEGER.
                 On entry,  M  specifies  the number  of rows  of the  matrix

                 op( A )  and of the  matrix  C.  M  must  be at least  zero.

                 Unchanged on exit.

        N      - INTEGER.
                 On entry,  N  specifies the number  of columns of the matrix

                 op( B ) and the number of columns of the matrix C. N must be

                 at least zero.
                 Unchanged on exit.

        K      - INTEGER.
                 On entry,  K  specifies  the number of columns of the matrix

                 op( A ) and the number of rows of the matrix op( B ). K must

                 be at least  zero.
                 Unchanged on exit.

        ALPHA  - DOUBLE PRECISION.
                 On entry, ALPHA specifies the scalar alpha.
                 Unchanged on exit.

        A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is

                 k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
                 Before entry with  TRANSA = 'N' or 'n',  the leading  m by k

                 part of the array  A  must contain the matrix  A,  otherwise

                 the leading  k by m  part of the array  A  must contain  the

                 matrix A.
                 Unchanged on exit.

        LDA    - INTEGER.
                 On entry, LDA specifies the first dimension of A as declared

                 in the calling (sub) program. When  TRANSA = 'N' or 'n' then

                 LDA must be at least  max( 1, m ), otherwise  LDA must be at

                 least  max( 1, k ).
                 Unchanged on exit.

        B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is

                 n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
                 Before entry with  TRANSB = 'N' or 'n',  the leading  k by n

                 part of the array  B  must contain the matrix  B,  otherwise

                 the leading  n by k  part of the array  B  must contain  the

                 matrix B.
                 Unchanged on exit.

        LDB    - INTEGER.
                 On entry, LDB specifies the first dimension of B as declared

                 in the calling (sub) program. When  TRANSB = 'N' or 'n' then

                 LDB must be at least  max( 1, k ), otherwise  LDB must be at

                 least  max( 1, n ).
                 Unchanged on exit.

        BETA   - DOUBLE PRECISION.
                 On entry,  BETA  specifies the scalar  beta.  When  BETA  is

                 supplied as zero then C need not be set on input.
                 Unchanged on exit.

        C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
                 Before entry, the leading  m by n  part of the array  C must

                 contain the matrix  C,  except when  beta  is zero, in which

                 case C need not be set on entry.
                 On exit, the array  C  is overwritten by the  m by n  matrix

                 ( alpha*op( A )*op( B ) + beta*C ).

        LDC    - INTEGER.
                 On entry, LDC specifies the first dimension of C as declared

                 in  the  calling  (sub)  program.   LDC  must  be  at  least

                 max( 1, m ).
                 Unchanged on exit.


        Level 3 Blas routine.

        -- Written on 8-February-1989.
           Jack Dongarra, Argonne National Laboratory.
           Iain Duff, AERE Harwell.
           Jeremy Du Croz, Numerical Algorithms Group Ltd.
           Sven Hammarling, Numerical Algorithms Group Ltd.



           Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not

           transposed and set  NROWA, NCOLA and  NROWB  as the number of rows

           and  columns of  A  and the  number of  rows  of  B  respectively.



       Parameter adjustments
           Function Body */

    //#define A(I, J) a[(I)-1 + ((J)-1) * (*lda)]
    // #define B(I, J) b[(I)-1 + ((J)-1) * (*ldb)]
    // #define C(I, J) c[(I)-1 + ((J)-1) * (*ldc)]

    auto A = [&](int I, int J) -> const double & {
        println("A({}, {}) = A[{}] = {}", I - 1, J - 1, (I)-1 + ((J)-1) * (*lda), a[(I)-1 + ((J)-1) * (*lda)]);
        return a[(I)-1 + ((J)-1) * (*lda)];
    };
    auto B = [&](int I, int J) -> const double & {
        println("B({}, {}) = B[{}] = {}", I - 1, J - 1, (I)-1 + ((J)-1) * (*ldb), b[(I)-1 + ((J)-1) * (*ldb)]);
        return b[(I)-1 + ((J)-1) * (*ldb)];
    };
    auto C = [&](int I, int J) -> double & {
        println("C({}, {}) = A[{}] = {}", I - 1, J - 1, (I)-1 + ((J)-1) * (*ldc), c[(I)-1 + ((J)-1) * (*ldc)]);
        return c[(I)-1 + ((J)-1) * (*ldc)];
    };

    nota = lsame(transa, "N");
    notb = lsame(transb, "N");
    if (nota) {
        nrowa = *m;
        ncola = *k;
    } else {
        nrowa = *k;
        ncola = *m;
    }
    if (notb) {
        nrowb = *k;
    } else {
        nrowb = *n;
    }

    /*     Test the input parameters. */

    info = 0;
    if (!nota && !lsame(transa, "C") && !lsame(transa, "T")) {
        info = 1;
    } else if (!notb && !lsame(transb, "C") && !lsame(transb, "T")) {
        info = 2;
    } else if (*m < 0) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*k < 0) {
        info = 5;
    } else if (*lda < std::max(1, nrowa)) {
        info = 8;
    } else if (*ldb < std::max(1, nrowb)) {
        info = 10;
    } else if (*ldc < std::max(1, *m)) {
        info = 13;
    }
    if (info != 0) {
        xerbla("DGEMM ", &info);
        return 0;
    }

    /*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
        return 0;
    }

    /*     And if  alpha.eq.zero. */

    if (*alpha == 0.) {
        if (*beta == 0.) {
            i__1 = *n;
            for (j = 1; j <= *n; ++j) {
                i__2 = *m;
                for (i = 1; i <= *m; ++i) {
                    C(i, j) = 0.;
                    /* L10: */
                }
                /* L20: */
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= *n; ++j) {
                i__2 = *m;
                for (i = 1; i <= *m; ++i) {
                    C(i, j) = *beta * C(i, j);
                    /* L30: */
                }
                /* L40: */
            }
        }
        return 0;
    }

    /*     Start the operations. */

    if (notb) {
        if (nota) {

            /*           Form  C := alpha*A*B + beta*C. */

            i__1 = *n;
            for (j = 1; j <= *n; ++j) {
                if (*beta == 0.) {
                    i__2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i, j) = 0.;
                        /* L50: */
                    }
                } else if (*beta != 1.) {
                    i__2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i, j) = *beta * C(i, j);
                        /* L60: */
                    }
                }
                i__2 = *k;
                for (l = 1; l <= *k; ++l) {
                    if (B(l, j) != 0.) {
                        temp = *alpha * B(l, j);
                        i__3 = *m;
                        for (i = 1; i <= *m; ++i) {
                            C(i, j) += temp * A(i, l);
                            /* L70: */
                        }
                    }
                    /* L80: */
                }
                /* L90: */
            }
        } else {

            /*           Form  C := alpha*A'*B + beta*C */

            i__1 = *n;
            for (j = 1; j <= *n; ++j) {
                i__2 = *m;
                for (i = 1; i <= *m; ++i) {
                    temp = 0.;
                    i__3 = *k;
                    for (l = 1; l <= *k; ++l) {
                        temp += A(l, i) * B(l, j);
                        /* L100: */
                    }
                    if (*beta == 0.) {
                        C(i, j) = *alpha * temp;
                    } else {
                        C(i, j) = *alpha * temp + *beta * C(i, j);
                    }
                    /* L110: */
                }
                /* L120: */
            }
        }
    } else {
        if (nota) {

            /*           Form  C := alpha*A*B' + beta*C */

            i__1 = *n;
            for (j = 1; j <= *n; ++j) {
                if (*beta == 0.) {
                    i__2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i, j) = 0.;
                        /* L130: */
                    }
                } else if (*beta != 1.) {
                    i__2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i, j) = *beta * C(i, j);
                        /* L140: */
                    }
                }
                i__2 = *k;
                for (l = 1; l <= *k; ++l) {
                    if (B(j, l) != 0.) {
                        temp = *alpha * B(j, l);
                        i__3 = *m;
                        for (i = 1; i <= *m; ++i) {
                            C(i, j) += temp * A(i, l);
                            /* L150: */
                        }
                    }
                    /* L160: */
                }
                /* L170: */
            }
        } else {

            /*           Form  C := alpha*A'*B' + beta*C */

            i__1 = *n;
            for (j = 1; j <= *n; ++j) {
                i__2 = *m;
                for (i = 1; i <= *m; ++i) {
                    temp = 0.;
                    i__3 = *k;
                    for (l = 1; l <= *k; ++l) {
                        temp += A(l, i) * B(j, l);
                        /* L180: */
                    }
                    if (*beta == 0.) {
                        C(i, j) = *alpha * temp;
                    } else {
                        C(i, j) = *alpha * temp + *beta * C(i, j);
                    }
                    /* L190: */
                }
                /* L200: */
            }
        }
    }

    return 0;

    /*     End of DGEMM . */

} /* dgemm_ */
} // namespace

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc) {
    _dgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

} // namespace einsums::backend::netlib