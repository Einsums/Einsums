#include "scale.h"

#include "util/macros.h"
#include "internal/1v/scale.hpp"
#include "internal/1v/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_vector_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_vector* A)
{
    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(
        [&](const communicator& comm)
        {
            if (A->alpha<T>() == T(0))
            {
                internal::set<T>(comm, get_config(cfg), A->n,
                                 T(0), static_cast<T*>(A->data), A->inc);
            }
            else if (A->alpha<T>() != T(1) || (is_complex<T>::value && A->conj))
            {
                internal::scale<T>(comm, get_config(cfg), A->n,
                                   A->alpha<T>(), A->conj,
                                   static_cast<T*>(A->data), A->inc);
            }
        }, comm);

        A->alpha<T>() = T(1);
        A->conj = false;
    })
}

}

}
