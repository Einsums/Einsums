//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime.hpp>

int einsums_main() {
    EINSUMS_PROFILE_SCOPE("einsums_main");

    using namespace einsums;

    // auto performance_counter = profile::detail::PerformanceCounter::create();
    // auto nevents             = performance_counter->nevents();
    // auto event_names         = performance_counter->event_names();

    // std::vector<uint64_t> start(nevents);
    // std::vector<uint64_t> stop(nevents);

    size_t i{10};
    auto   A = create_random_tensor("A", i);
    auto   B = create_random_tensor("B", i);

    // performance_counter->start(start);
    double C = linear_algebra::dot(A, B);
    // performance_counter->stop(stop);

    println(A);
    println(B);
    println("C = {}", C);

    // performance_counter->delta(start, stop);

    // println("events: {}", event_names);
    // println("performance: {}", stop);

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
