#include "util/cpuid.hpp"
#include "config.hpp"

namespace tblis
{

int knl_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {
        if (get_verbose() >= 1) printf("tblis: knl: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA3))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support FMA3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX2))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX2.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX512F))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX512F.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX512PF))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX512PF.\n");
        return -1;
    }

    return 4;
}

}
