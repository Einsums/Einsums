#include "scale.h"

#include "util/macros.h"
#include "internal/1m/scale.hpp"
#include "internal/1m/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_matrix_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_matrix* A)
{
    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(
        [&](const communicator& comm)
        {
           if (A->alpha<T>() == T(0))
           {
               internal::set<T>(comm, get_config(cfg), A->m, A->n,
                                T(0), static_cast<T*>(A->data), A->rs, A->cs);
           }
           else if (A->alpha<T>() != T(1) || (is_complex<T>::value && A->conj))
           {
               internal::scale<T>(comm, get_config(cfg), A->m, A->n,
                                  A->alpha<T>(), A->conj,
                                  static_cast<T*>(A->data), A->rs, A->cs);
           }
        }, comm);

        A->alpha<T>() = T(1);
        A->conj = false;
    })
}

}

}
