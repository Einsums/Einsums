#include "thread.h"

#include <cstdio>

#if TBLIS_HAVE_SYSCTL
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if TBLIS_HAVE_SYSCONF
#include <unistd.h>
#endif

#if TBLIS_HAVE_HWLOC_H
#include <hwloc.h>
#endif

const tblis_comm* const tblis_single = tci_single;

namespace
{

struct thread_configuration
{
    unsigned num_threads = 1;

    thread_configuration()
    {
        const char* str = nullptr;

#if !TBLIS_ENABLE_TBB
        str = getenv("TBLIS_NUM_THREADS");
        if (!str) str = getenv("OMP_NUM_THREADS");
#endif //!TBLIS_ENABLE_TBB

        if (str)
        {
            num_threads = strtol(str, NULL, 10);
        }
        else
        {
            #if TBLIS_HAVE_HWLOC_H

            hwloc_topology_t topo;
            hwloc_topology_init(&topo);
            hwloc_topology_load(topo);

            int depth = hwloc_get_cache_type_depth(topo, 1, HWLOC_OBJ_CACHE_DATA);
            if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
            {
                num_threads = hwloc_get_nbobjs_by_depth(topo, depth);
                printf("nt: %d\n", num_threads);
            }

            hwloc_topology_destroy(topo);

            #elif TBLIS_HAVE_LSCPU

            FILE *fd = popen("lscpu --parse=core | grep '^[0-9]' | sort -rn | head -n 1", "r");

            std::string s;
            int c;
            while ((c = fgetc(fd)) != EOF) s.push_back(c+1);

            pclose(fd);

            num_threads = strtol(s.c_str(), NULL, 10);

            #elif TBLIS_HAVE_SYSCTLBYNAME

            size_t len = sizeof(num_threads);
            sysctlbyname("hw.physicalcpu", &num_threads, &len, NULL, 0);

            #elif TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_ONLN

            num_threads = sysconf(_SC_NPROCESSORS_ONLN);

            #elif TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_CONF

            num_threads = sysconf(_SC_NPROCESSORS_CONF);

            #endif
        }
    }
};

thread_configuration& get_thread_configuration()
{
    static thread_configuration cfg;
    return cfg;
}

}

namespace tblis
{

tci::communicator single;

std::atomic<long> flops{0};
len_type inout_ratio = 200000;

}

extern "C"
{

unsigned tblis_get_num_threads()
{
    return get_thread_configuration().num_threads;
}

void tblis_set_num_threads(unsigned num_threads)
{
    get_thread_configuration().num_threads = num_threads;
}

}
