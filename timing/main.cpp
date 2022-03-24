#include "einsums/LinearAlgebra.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/State.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/Timer.hpp"

auto main() -> int {
    ////////////////////////////////////
    // Form the two-electron integrals//
    ////////////////////////////////////
    using namespace einsums;
    using namespace TensorAlgebra;
    using namespace TensorAlgebra::Index;

    Timer::initialize();

#define NMO 10
#define NBS 10

    int nmo1{NMO}, nmo2{NMO}, nmo3{NMO}, nmo4{NMO};
    int nbs1{NBS}, nbs2{NBS}, nbs3{NBS}, nbs4{NBS};

    println("Running on {} threads", omp_get_max_threads());
    println("NMO {} :: NBS {}", NMO, NBS);

    Timer::push("Allocations");
    auto GAO = std::make_unique<Tensor<4>>("AOs", nbs1, nbs2, nbs3, nbs4);
    Tensor<2> C1{"C1", nbs1, nmo1};
    Tensor<2> C2{"C2", nbs2, nmo2};
    Tensor<2> C3{"C3", nbs3, nmo3};
    Tensor<2> C4{"C4", nbs4, nmo4};
    Timer::pop();

    Timer::push("Full Transformation");

    // Transform ERI AO Tensor to ERI MO Tensor
    Timer::push("C4");
    Timer::push("Allocation 1");
    auto pqrS = std::make_unique<Tensor<4>>("pqrS", nbs1, nbs2, nbs3, nmo4);
    Timer::pop();
    einsum(Indices{p, q, r, S}, &pqrS, Indices{p, q, r, s}, GAO, Indices{s, S}, C4);
    GAO.reset(nullptr);
    Timer::pop();

    Timer::push("C3");
    Timer::push("Allocation 1");
    auto pqSr = std::make_unique<Tensor<4>>("pqSr", nbs1, nbs2, nmo4, nbs3);
    Timer::pop();
    Timer::push("presort");
    sort(Indices{p, q, S, r}, &pqSr, Indices{p, q, r, S}, pqrS);
    Timer::pop();
    pqrS.reset(nullptr);

    Timer::push("Allocation 2");
    auto pqSR = std::make_unique<Tensor<4>>("pqSR", nbs1, nbs2, nmo4, nmo3);
    Timer::pop();
    einsum(Indices{p, q, S, R}, &pqSR, Indices{p, q, S, r}, pqSr, Indices{r, R}, C3);
    pqSr.reset(nullptr);
    Timer::pop();

    Timer::push("C2");
    Timer::push("Allocation 1");
    auto RSpq = std::make_unique<Tensor<4>>("RSpq", nmo3, nmo4, nbs1, nbs2);
    Timer::pop();
    Timer::push("presort");
    sort(Indices{R, S, p, q}, &RSpq, Indices{p, q, S, R}, pqSR);
    pqSR.reset(nullptr);
    Timer::pop();

    Timer::push("Allocation 2");
    auto RSpQ = std::make_unique<Tensor<4>>("RSpQ", nmo3, nmo4, nbs1, nmo2);
    Timer::pop();
    einsum(Indices{R, S, p, Q}, &RSpQ, Indices{R, S, p, q}, RSpq, Indices{q, Q}, C2);
    RSpq.reset(nullptr);
    Timer::pop();

    Timer::push("C1");
    Timer::push("Allocation 1");
    auto RSQp = std::make_unique<Tensor<4>>("RSQp", nmo3, nmo4, nmo2, nbs1);
    Timer::pop();
    Timer::push("presort");
    sort(Indices{R, S, Q, p}, &RSQp, Indices{R, S, p, Q}, RSpQ);
    RSpQ.reset(nullptr);
    Timer::pop();

    Timer::push("Allocation 2");
    auto RSQP = std::make_unique<Tensor<4>>("RSQP", nmo3, nmo4, nmo2, nmo1);
    Timer::pop();
    einsum(Indices{R, S, Q, P}, &RSQP, Indices{R, S, Q, p}, RSQp, Indices{p, P}, C1);
    RSQp.reset(nullptr);
    Timer::pop();

    Timer::push("Sort RSQP -> PQRS");
    Timer::push("Allocation");
    Tensor<4> PQRS{"PQRS", nmo1, nmo2, nmo3, nmo4};
    Timer::pop();
    sort(Indices{P, Q, R, S}, &PQRS, Indices{R, S, Q, P}, RSQP);
    RSQP.reset(nullptr);
    Timer::pop();

    Timer::pop(); // Full Transformation

    element_transform(&PQRS, [](double value) -> double { return 1.0 / value; });

    Print::stacktrace();

    Timer::report();
    Timer::finalize();

    // Typically you would build a new wavefunction and populate it with data
    return EXIT_SUCCESS;
}
