#include "config.hpp"

#include "util/cpuid.hpp"

extern int vpu_count();

namespace tblis
{

int skx2_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA3))
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Doesn't support FMA3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX2))
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Doesn't support AVX2.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX512F))
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Doesn't support AVX512F.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX512DQ))
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Doesn't support AVX512DQ.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX512BW))
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Doesn't support AVX512BW.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX512VL))
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Doesn't support AVX512VL.\n");
        return -1;
    }

    int nvpu = vpu_count();
    if (nvpu != 2)
    {
        if (get_verbose() >= 1) printf("tblis: skx2: Wrong number of VPUs (%d).\n", nvpu);
        return -1;
    }

    return 4;
}

}
