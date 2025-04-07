#include "add.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg, len_type m, len_type n,
         T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
         T  beta, bool conj_B,       T* B, stride_type rs_B, stride_type cs_B)
{
    if (rs_B > cs_B)
    {
        std::swap(m, n);
        std::swap(rs_A, cs_A);
        std::swap(rs_B, cs_B);
    }

    /*
     * If A is row-major and B is column-major or vice versa, use
     * the transpose microkernel.
     */
    if (rs_A > cs_A)
    {
        const len_type MR = cfg.trans_mr.def<T>();
        const len_type NR = cfg.trans_nr.def<T>();

        comm.distribute_over_threads({m, MR}, {n, NR},
        [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
        {
            for (len_type i = m_min;i < m_max;i += MR)
            {
                len_type m_loc = std::min(m_max-i, MR);
                for (len_type j = n_min;j < n_max;j += NR)
                {
                    len_type n_loc = std::min(n_max-j, NR);
                    cfg.trans_ukr.call<T>(m_loc, n_loc,
                        alpha, conj_A, A + i*rs_A + j*cs_A, rs_A, cs_A,
                         beta, conj_B, B + i*rs_B + j*cs_B, rs_B, cs_B);
                }
            }
        });
    }
    /*
     * Otherwise, A can be added to B column-by-column or row-by-row
     */
    else
    {
        comm.distribute_over_threads(m, n,
        [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                cfg.add_ukr.call<T>(m_max-m_min,
                    alpha, conj_A, A + m_min*rs_A + j*cs_A, rs_A,
                     beta, conj_B, B + m_min*rs_B + j*cs_B, rs_B);
            }
        });
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, len_type m, len_type n, \
                  T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                  T  beta, bool conj_B,       T* B, stride_type rs_B, stride_type cs_B);
#include "configs/foreach_type.h"

}
}
