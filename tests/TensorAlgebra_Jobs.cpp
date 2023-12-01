
#include "einsums/TensorAlgebra.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/STL.hpp"
#include "einsums/Sort.hpp"
#include "einsums/State.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/Jobs.hpp"
#include "einsums/jobs/Job.hpp"

#include <H5Fpublic.h>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <complex>
#include <cstdio>
#include <thread>
#include <type_traits>
#include <stdexcept>
#include <exception>

TEST_CASE("Identity Tensor Locks", "[jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::jobs;

    Resource<Tensor<double, 2>> I_res(create_identity_tensor("I", 3, 3));

    auto lock = I_res.read_promise();

    lock->wait();

    auto I = lock->get();

    REQUIRE((*I)(0, 0) == 1.0);
    REQUIRE((*I)(0, 1) == 0.0);
    REQUIRE((*I)(0, 2) == 0.0);
    REQUIRE((*I)(1, 0) == 0.0);
    REQUIRE((*I)(1, 1) == 1.0);
    REQUIRE((*I)(1, 2) == 0.0);
    REQUIRE((*I)(2, 0) == 0.0);
    REQUIRE((*I)(2, 1) == 0.0);
    REQUIRE((*I)(2, 2) == 1.0);

    lock->release();
}

TEST_CASE("Scale Row Locks", "[jobs]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;
    using namespace einsums::jobs;

    Resource<Tensor<double, 2>> res(create_random_tensor("I", 3, 3));
    Resource<Tensor<double, 2>>     res_copy(*(res.get_data()));

    auto lock      = res.read_promise();
    auto lock_copy = res_copy.write_promise();

    while (!lock_copy->ready()) {
        std::this_thread::yield();
    }

    scale_row(1, 2.0, lock_copy->get().get());

    while (!lock->ready()) {
        std::this_thread::yield();
    }

    auto I_copy     = *(lock_copy->get());
    auto I_original = *(lock->get());

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == 2.0 * I_original(1, 0));
    REQUIRE(I_copy(1, 1) == 2.0 * I_original(1, 1));
    REQUIRE(I_copy(1, 2) == 2.0 * I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == I_original(2, 1));
    REQUIRE(I_copy(2, 2) == I_original(2, 2));

    lock_copy->release();
    lock->release();
}

TEST_CASE("Scale Column Locks", "[jobs]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;
    using namespace einsums::jobs;

    Resource<Tensor<double, 2>> res(create_random_tensor("I", 3, 3));
    Resource<Tensor<double, 2>>     res_copy(*(res.get_data()));

    auto lock      = res.read_promise();
    auto lock_copy = res_copy.write_promise();

    while (!lock_copy->ready()) {
        std::this_thread::yield();
    }

    scale_column(1, 2.0, lock_copy->get().get());

    while (!lock->ready()) {
        std::this_thread::yield();
    }

    auto I_copy     = *(lock_copy->get());
    auto I_original = *(lock->get());

    REQUIRE(I_copy(0, 0) == I_original(0, 0));
    REQUIRE(I_copy(0, 1) == 2.0 * I_original(0, 1));
    REQUIRE(I_copy(0, 2) == I_original(0, 2));
    REQUIRE(I_copy(1, 0) == I_original(1, 0));
    REQUIRE(I_copy(1, 1) == 2.0 * I_original(1, 1));
    REQUIRE(I_copy(1, 2) == I_original(1, 2));
    REQUIRE(I_copy(2, 0) == I_original(2, 0));
    REQUIRE(I_copy(2, 1) == 2.0 * I_original(2, 1));
    REQUIRE(I_copy(2, 2) == I_original(2, 2));

    lock->release();
    lock_copy->release();
}

TEST_CASE("einsum1 job", "[jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    SECTION("ik=ij,jk") {
	std::shared_ptr<jobs::Resource<Tensor<double, 2>>> A_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("A", 3, 3),
	  B_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("B", 3, 3),
	  C_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("C", 3, 3);

        // Initialize thread pool.
        INFO("Initialize the thread pool");
        jobs::ThreadPool::init(8);
        jobs::ThreadPool::get_singleton();

        // Start the job manager.
        INFO("Start the job manager");
        jobs::JobManager::get_singleton().start_manager();

        // Obtain locks for modification.
        INFO("Obtain locks for modification.");
        auto A_lock = A_res->write_promise();
        auto B_lock = B_res->write_promise();

        // Make and queue the job.
        INFO("Make a job and add it to the queue.");
        auto job = jobs::einsum(Indices{index::i, index::j}, C_res, Indices{index::i, index::k}, A_res, Indices{index::k, index::j}, B_res);

        // Wait for the modification locks to resolve.
        A_lock->wait();
        B_lock->wait();

        auto A = A_lock->get();
        auto B = B_lock->get();

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                (*A)(i, j) = ij;
                (*B)(i, j) = ij;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));

        A_lock->release();
        B_lock->release();

        while (!job.lock()->get_state() == einsums::jobs::detail::FINISHED) {
            std::this_thread::yield();
        }

        auto C_lock = C_res->read_promise();

        C_lock->wait();

        auto C = C_lock->get();

        // println(A);
        // println(B);
        // println(C);

        /*[[ 30,  36,  42],
           [ 66,  81,  96],
           [102, 126, 150]]*/
        REQUIRE((*C)(0, 0) == 30.0);
        REQUIRE((*C)(0, 1) == 36.0);
        REQUIRE((*C)(0, 2) == 42.0);
        REQUIRE((*C)(1, 0) == 66.0);
        REQUIRE((*C)(1, 1) == 81.0);
        REQUIRE((*C)(1, 2) == 96.0);
        REQUIRE((*C)(2, 0) == 102.0);
        REQUIRE((*C)(2, 1) == 126.0);
        REQUIRE((*C)(2, 2) == 150.0);
        INFO("Releasing memory");
        C_lock->release();

        jobs::JobManager::get_singleton().wait_on_jobs();
        jobs::ThreadPool::get_singleton().wait_on_running();
        INFO("Killing the manager");
        jobs::JobManager::get_singleton().stop_manager();
        INFO("Destroying the thread pool.");
        jobs::ThreadPool::destroy();
        jobs::JobManager::destroy();

        INFO("Deleting tensors.");
    }

    SECTION("il=ijk,jkl") {
        std::shared_ptr<jobs::Resource<Tensor<double, 3>>> A_res = std::make_shared<jobs::Resource<Tensor<double, 3>>>("A", 3, 3, 3),
	    B_res = std::make_shared<jobs::Resource<Tensor<double, 3>>>("B", 3, 3, 3);
	std::shared_ptr<jobs::Resource<Tensor<double, 2>>> C_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("C", 3, 3);

	auto A_lock = A_res->write_promise();
	auto B_lock = B_res->write_promise();

	jobs::ThreadPool::init(8);
	jobs::JobManager::get_singleton().start_manager();

	auto ein_job = jobs::einsum(Indices{index::i, index::l}, C_res, Indices{index::i, index::j, index::k}, A_res,
	    Indices{index::j, index::k, index::l}, B_res);

	auto A = A_lock->get();
	auto B = B_lock->get();
	
        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    (*A)(i, j, k) = ij;
                    (*B)(i, j, k) = ij;
                }
            }
        }

        auto C_lock = C_res->read_promise();
	
	A_lock->release();
	B_lock->release();

        auto C = C_lock->get();

        // array([[ 765.,  810.,  855.],
        //        [1818., 1944., 2070.],
        //        [2871., 3078., 3285.]])
        REQUIRE((*C)(0, 0) == 765.0);
        REQUIRE((*C)(0, 1) == 810.0);
        REQUIRE((*C)(0, 2) == 855.0);
        REQUIRE((*C)(1, 0) == 1818.0);
        REQUIRE((*C)(1, 1) == 1944.0);
        REQUIRE((*C)(1, 2) == 2070.0);
        REQUIRE((*C)(2, 0) == 2871.0);
        REQUIRE((*C)(2, 1) == 3078.0);
        REQUIRE((*C)(2, 2) == 3285.0);

	C_lock->release();

    jobs::JobManager::get_singleton().wait_on_jobs();
    jobs::ThreadPool::get_singleton().wait_on_running();
	jobs::JobManager::get_singleton().stop_manager();
	jobs::ThreadPool::destroy();
	jobs::JobManager::destroy();
    }
}

TEST_CASE("einsum2 jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    SECTION("3x3 <- 3x5 * 5x3") {
        Tensor C0{"C0", 3, 3};
        Tensor C1{"C1", 3, 3};
        Tensor A = create_random_tensor("A", 3, 5);
        Tensor B = create_random_tensor("B", 5, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3x3 <- 3x5 * 3x5") {
        Tensor C0{"C0", 3, 3};
        Tensor C1{"C1", 3, 3};
        Tensor A = create_random_tensor("A", 3, 5);
        Tensor B = create_random_tensor("B", 3, 5);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{j, k}, B));
        linear_algebra::gemm<false, true>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3 <- 3x5 * 5") {
        Tensor C0{"C0", 3};
        Tensor C1{"C1", 3};
        Tensor A = create_random_tensor("A", 3, 5);
        Tensor B = create_random_tensor("B", 5);

        C0.zero();
        C1.zero();

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j}, A, Indices{j}, B));
        linear_algebra::gemv<false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            REQUIRE_THAT(C0(i0), Catch::Matchers::WithinAbs(C1(i0), 0.001));
        }
    }

    SECTION("3 <- 3x4x5 * 4x3x5") {
        Tensor C0{"C0", 3};
        zero(C0);
        Tensor C1{"C1", 3};
        zero(C1);
        Tensor A = create_random_tensor("A", 3, 4, 5);
        Tensor B = create_random_tensor("B", 4, 3, 5);

        REQUIRE_NOTHROW(einsum(Indices{i}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));

        for (size_t i0 = 0; i0 < 3; i0++) {
            double sum{0};

            for (size_t j0 = 0; j0 < 4; j0++) {
                for (size_t k0 = 0; k0 < 5; k0++) {
                    sum += A(i0, j0, k0) * B(j0, i0, k0);
                }
            }
            C1(i0) = sum;
        }

        for (size_t i0 = 0; i0 < 3; i0++) {
            REQUIRE_THAT(C0(i0), Catch::Matchers::WithinRel(C1(i0), 0.0001));
        }
    }

    SECTION("3x5 <- 3x4x5 * 4x3x5") {
        Tensor C0{"C0", 3, 5};
        Tensor C1{"C1", 3, 5};
        zero(C0);
        zero(C1);
        Tensor A = create_random_tensor("A", 3, 4, 5);
        Tensor B = create_random_tensor("B", 4, 3, 5);

        // timer::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, k}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // timer::pop();

        // timer::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                double sum{0};
                for (size_t j0 = 0; j0 < 4; j0++) {

                    sum += A(i0, j0, k0) * B(j0, i0, k0);
                }
                C1(i0, k0) = sum;
            }
        }
        // timer::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3, l <- 3x4x5 * 4x3x5") {
        Tensor C0{"C0", 3, 5};
        zero(C0);
        Tensor C1{"C1", 3, 5};
        zero(C1);
        Tensor A = create_random_tensor("A", 3, 4, 5);
        Tensor B = create_random_tensor("B", 4, 3, 5);

        // timer::push("einsum: 3x5 <- 3x4x5 * 4x3x5");
        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C0, Indices{i, j, k}, A, Indices{j, i, k}, B));
        // timer::pop();

        // timer::push("hand  : 3x5 <- 3x4x5 * 4x3x5");
        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t k0 = 0; k0 < 5; k0++) {
                for (size_t l0 = 0; l0 < 5; l0++) {
                    double sum{0};
                    for (size_t j0 = 0; j0 < 4; j0++) {

                        sum += A(i0, j0, k0) * B(j0, i0, k0);
                    }
                    C1(i0, l0) += sum;
                }
            }
        }
        // timer::pop();

        for (size_t i0 = 0; i0 < 3; i0++) {
            for (size_t j0 = 0; j0 < 5; j0++) {
                // REQUIRE(C0(i0, j0) == C1(i0, j0));?
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0000001));
            }
        }
    }

    // timer::report();
    // timer::finalize();
}

TEST_CASE("einsum3 jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // timer::initialize();

    SECTION("3x3 <- 3x5 * 5x3") {
        Tensor C0{"C0", 3, 3};
        Tensor C1{"C1", 3, 3};
        Tensor A = create_random_tensor("A", 3, 5);
        Tensor B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("3x3x3x3 <- 3x3x3x3 * 3x3") {
        // This one is to represent a two-electron integral transformation
        Tensor gMO0{"g0", 3, 3, 3, 3};
        Tensor gMO1{"g1", 3, 3, 3, 3};
        zero(gMO0);
        zero(gMO1);
        Tensor A = create_random_tensor("A", 3, 3, 3, 3);
        Tensor B = create_random_tensor("B", 3, 3);

        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &gMO0, Indices{i, j, k, p}, A, Indices{p, l}, B));

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, k0, p0) * B(p0, l0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &gMO0, Indices{i, j, p, l}, A, Indices{p, k}, B));

        gMO1.zero();
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, p0, l0) * B(p0, k0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                auto vgMO0 = gMO0(i0, j0, All, All);
                REQUIRE_NOTHROW(einsum(Indices{k, l}, &vgMO0, Indices{p, l}, A(i0, j0, All, All), Indices{p, k}, B));
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }
    }

    // timer::report();
    // timer::finalize();
}

TEST_CASE("einsum4 jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // timer::initialize();
    SECTION("3x3x3x3 <- 3x3x3x3 * 3x3") {
        // This one is to represent a two-electron integral transformation
        Tensor gMO0{"g0", 3, 3, 3, 3};
        Tensor gMO1{"g1", 3, 3, 3, 3};
        zero(gMO0);
        zero(gMO1);
        Tensor A = create_random_tensor("A", 3, 3, 3, 3);
        Tensor B = create_random_tensor("B", 3, 3);

        REQUIRE_NOTHROW(einsum(Indices{p, q, r, l}, &gMO0, Indices{p, q, r, s}, A, Indices{s, l}, B));

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, k0, p0) * B(p0, l0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{p, q, k, s}, &gMO0, Indices{p, q, r, s}, A, Indices{r, k}, B));

        gMO1.zero();
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        for (size_t p0 = 0; p0 < B.dim(0); p0++) {
                            gMO1(i0, j0, k0, l0) += A(i0, j0, p0, l0) * B(p0, k0);
                        }
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        REQUIRE_THAT(gMO0(i0, j0, k0, l0), Catch::Matchers::WithinAbs(gMO1(i0, j0, k0, l0), 0.001));
                    }
                }
            }
        }

#if 0
        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                auto vgMO0 = gMO0(i0, j0, All, All);
                TensorAlgebra::einsum(Indices{k, s}, &vgMO0, Indices{r, s}, A(i0, j0, All, All), Indices{r, k}, B);
            }
        }

        for (size_t i0 = 0; i0 < gMO0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < gMO0.dim(1); j0++) {
                for (size_t k0 = 0; k0 < gMO0.dim(2); k0++) {
                    for (size_t l0 = 0; l0 < gMO0.dim(3); l0++) {
                        // println("i0 %lu j0 %lu k0 %lu l0 %lu, gMO0 %lf, gMO1 %lf", i0, j0, k0, l0, gMO0(i0, j0, k0, l0),
                        // gMO1(i0, j0, k0, l0));
                        REQUIRE(gMO0(i0, j0, k0, l0) == gMO1(i0, j0, k0, l0));
                    }
                }
            }
        }
#endif
    }

    // timer::report();
    // timer::finalize();
}

TEST_CASE("IntegralTransformation jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // timer::initialize();
    // SECTION("3x3x3x3 <- 3x3x3x3 * 3x3 * 3x3 * 3x3 * 3x3") {
    //     Tensor C1 = create_random_tensor("C1", 4, 2);
    //     Tensor C2 = create_random_tensor("C2", 4, 2);
    //     Tensor C3 = create_random_tensor("C3", 4, 3);
    //     Tensor C4 = create_random_tensor("C4", 4, 4);

    //     Tensor true_answer{"true", 2, 2, 3, 4};
    //     Tensor memory_ao = create_random_tensor("ao", 4, 4, 4, 4);
    //     DiskTensor disk_ao{State::data, "ao", 4, 4, 4, 4};

    //     // #pragma omp parallel for collapse(8)
    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {

    //                     for (size_t p0 = 0; p0 < C1.dim(0); p0++) {
    //                         for (size_t q0 = 0; q0 < C2.dim(0); q0++) {
    //                             for (size_t r0 = 0; r0 < C3.dim(0); r0++) {
    //                                 for (size_t s0 = 0; s0 < C4.dim(0); s0++) {
    //                                     true_answer(i0, j0, k0, l0) +=
    //                                         C1(p0, i0) * C2(q0, j0) * C3(r0, k0) * C4(s0, l0) * memory_ao(p0, q0, r0, s0);
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // Save our inital memory_ao to disk
    //     write(State::data, C1);
    //     write(State::data, C2);
    //     write(State::data, C3);
    //     write(State::data, C4);

    //     disk_ao(All, All, All, All) = memory_ao;
    //     auto temp = disk_ao(All, All, All, All);
    //     auto &temp2 = temp.get();

    //     // Ensure the data was saved to disk correctly
    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {
    //                     REQUIRE(temp2(i0, j0, k0, l0) == memory_ao(i0, j0, k0, l0));
    //                 }
    //             }
    //         }
    //     }

    //     auto memory_result = transformation("mo", memory_ao, C1, C2, C3, C4);
    //     auto disk_result = transformation("mo", disk_ao, C1, C2, C3, C4);

    //     // Make sure the memory and disk results match
    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {
    //                     REQUIRE_THAT(memory_result(i0, j0, k0, l0), Catch::Matchers::WithinRel(true_answer(i0, j0, k0, l0), 0.0000001));
    //                 }
    //             }
    //         }
    //     }

    //     for (size_t i0 = 0; i0 < C1.dim(1); i0++) {
    //         for (size_t j0 = 0; j0 < C2.dim(1); j0++) {
    //             auto disk_view = disk_result(i0, j0, All, All);
    //             auto &disk_tensor = disk_view.get();
    //             for (size_t k0 = 0; k0 < C3.dim(1); k0++) {
    //                 for (size_t l0 = 0; l0 < C4.dim(1); l0++) {
    //                     REQUIRE_THAT(disk_tensor(k0, l0), Catch::Matchers::WithinRel(true_answer(i0, j0, k0, l0), 0.0000001));
    //                 }
    //             }
    //         }
    //     }
    // }

    SECTION("R2 <- R3 * R3") {
        Tensor W_mi = create_random_tensor("W_mi", 4, 4);
        Tensor g_m  = create_random_tensor("g_m", 4, 8, 8);
        Tensor t_i  = create_random_tensor("t_i", 4, 8, 8);

        // println(W_mi);
        // println(g_m);
        // println(t_i);

        REQUIRE_NOTHROW(einsum(1.0, Indices{index::n, index::j}, &W_mi, 0.25, Indices{index::n, index::e, index::f}, g_m,
                               Indices{index::j, index::e, index::f}, t_i));
    }
}

TEST_CASE("Hadamard jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- i,i * j*j") {
        Tensor A = create_random_tensor("A", _i, _i);
        Tensor B = create_random_tensor("B", _j, _j);
        Tensor C{"C", _i, _j};
        Tensor C0{"C0", _i, _j};
        C0.zero();
        C.zero();

        // println(A);
        // println(B);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, i0) * B(j0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, i}, A, Indices{j, j}, B));

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j <- i,i,j * j,j,i") {
        Tensor A = create_random_tensor("A", _i, _i, _j);
        Tensor B = create_random_tensor("B", _j, _j, _i);
        Tensor C{"C", _i, _j};
        Tensor C0{"C0", _i, _j};
        C0.zero();
        C.zero();

        // println(A);
        // println(B);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, i0, j0) * B(j0, j0, i0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, i, j}, A, Indices{j, j, i}, B));

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j <- i,j,i * j,i,j") {
        Tensor A = create_random_tensor("A", _i, _j, _i);
        Tensor B = create_random_tensor("B", _j, _i, _j);
        Tensor C{"C", _i, _j};
        Tensor C0{"C0", _i, _j};
        C0.zero();
        C.zero();

        // println(A);
        // println(B);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.0001));
            }
        }
    }

    SECTION("i,j,i <- i,j,i * j,i,j") {
        Tensor A = create_random_tensor("A", _i, _j, _i);
        Tensor B = create_random_tensor("B", _j, _i, _j);
        Tensor C{"C", _i, _j, _i};
        Tensor C0{"C0", _i, _j, _i};
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CHECK_THAT(C(i0, j0, i0), Catch::Matchers::WithinRel(C0(i0, j0, i0), 0.00001));
            }
        }
    }

    SECTION("i,i,i <- i,j,i * j,i,j") {
        Tensor A = create_random_tensor("A", _i, _j, _i);
        Tensor B = create_random_tensor("B", _j, _i, _j);
        Tensor C{"C", _i, _i, _i};
        Tensor C0{"C0", _i, _i, _i};
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, i0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, i, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            CHECK_THAT(C(i0, i0, i0), Catch::Matchers::WithinRel(C0(i0, i0, i0), 0.00001));
        }
    }

    SECTION("i,i <- i,j,k * j,i,k") {
        Tensor A = create_random_tensor("A", _i, _j, _k);
        Tensor B = create_random_tensor("B", _j, _i, _k);
        Tensor C{"C", _i, _i};
        Tensor C0{"C0", _i, _i};
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, i0) += A(i0, j0, k0) * B(j0, i0, k0);
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, i}, &C, Indices{i, j, k}, A, Indices{j, i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _i; j0++) {
                CHECK_THAT(C(i0, j0), Catch::Matchers::WithinRel(C0(i0, j0), 0.00001));
            }
        }
    }
}

TEST_CASE("unique_ptr jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    SECTION("C") {
        auto   C0 = std::make_unique<Tensor<double, 2>>("C0", 3, 3);
        Tensor C1{"C1", 3, 3};
        Tensor A = create_random_tensor("A", 3, 5);
        Tensor B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("A") {
        Tensor C0{"C0", 3, 3};
        Tensor C1{"C1", 3, 3};
        auto   A = std::make_unique<Tensor<double, 2>>(create_random_tensor("A", 3, 5));
        Tensor B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("B") {
        Tensor C0{"C0", 3, 3};
        Tensor C1{"C1", 3, 3};
        Tensor A = create_random_tensor("A", 3, 5);
        auto   B = std::make_unique<Tensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("AB") {
        Tensor C0{"C0", 3, 3};
        Tensor C1{"C1", 3, 3};
        auto   A = std::make_unique<Tensor<double, 2>>(create_random_tensor("A", 3, 5));
        auto   B = std::make_unique<Tensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0.dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0.dim(1); j0++) {
                REQUIRE_THAT(C0(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CA") {
        auto   C0 = std::make_unique<Tensor<double, 2>>("C0", 3, 3);
        Tensor C1{"C1", 3, 3};
        auto   A = std::make_unique<Tensor<double, 2>>(create_random_tensor("A", 3, 5));
        Tensor B = create_random_tensor("B", 5, 3);

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CB") {
        auto   C0 = std::make_unique<Tensor<double, 2>>("C0", 3, 3);
        Tensor C1{"C1", 3, 3};
        Tensor A = create_random_tensor("A", 3, 5);
        auto   B = std::make_unique<Tensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }

    SECTION("CAB") {
        auto   C0 = std::make_unique<Tensor<double, 2>>("C0", 3, 3);
        Tensor C1{"C1", 3, 3};
        auto   A = std::make_unique<Tensor<double, 2>>(create_random_tensor("A", 3, 5));
        auto   B = std::make_unique<Tensor<double, 2>>(create_random_tensor("B", 5, 3));

        // Working to get the einsum to perform the gemm that follows.
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C0, Indices{i, k}, A, Indices{k, j}, B));
        linear_algebra::gemm<false, false>(1.0, *A, *B, 0.0, &C1);

        for (size_t i0 = 0; i0 < C0->dim(0); i0++) {
            for (size_t j0 = 0; j0 < C0->dim(1); j0++) {
                REQUIRE_THAT(C0->operator()(i0, j0), Catch::Matchers::WithinRel(C1(i0, j0), 0.0001));
            }
        }
    }
}

TEST_CASE("Transpose C jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- j,k * k,i === true, false, false") {
        Tensor A = create_random_tensor("A", _j, _k);
        Tensor B = create_random_tensor("B", _k, _i);
        Tensor C{"C", _i, _j};
        Tensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j, k}, A, Indices{k, i}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(j0, k0) * B(k0, i0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * k,i === true, true, false") {
        Tensor A = create_random_tensor("A", _k, _j);
        Tensor B = create_random_tensor("B", _k, _i);
        Tensor C{"C", _i, _j};
        Tensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{k, j}, A, Indices{k, i}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(k0, j0) * B(k0, i0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- j,k * i,k === true, false, true") {
        Tensor A = create_random_tensor("A", _j, _k);
        Tensor B = create_random_tensor("B", _i, _k);
        Tensor C{"C", _i, _j};
        Tensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j, k}, A, Indices{i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(j0, k0) * B(i0, k0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * i,k === true, true, true") {
        Tensor A = create_random_tensor("A", _k, _j);
        Tensor B = create_random_tensor("B", _i, _k);
        Tensor C{"C", _i, _j};
        Tensor C0{"C0", _i, _j};
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{k, j}, A, Indices{i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(k0, j0) * B(i0, k0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("Wmnij <- 0.25 t_ijef * g_mnef") {
        size_t _m = 12, _n = 12, _i = 5, _j = 5, _e = 7, _f = 7;

        Tensor Wmnij{"Wmnij", _m, _n, _i, _j};
        zero(Wmnij);
        Tensor W0{"Wmnij", _m, _n, _i, _j};
        zero(W0);

        Tensor t_oovv = create_random_tensor("t_oovv", _i, _j, _e, _f);
        Tensor g_oovv = create_random_tensor("g_oovv", _m, _n, _e, _f);

        REQUIRE_NOTHROW(einsum(1.0, Indices{m, n, i, j}, &Wmnij, 0.25, Indices{i, j, e, f}, t_oovv, Indices{m, n, e, f}, g_oovv));

        for (size_t m0 = 0; m0 < _m; m0++) {
            for (size_t n0 = 0; n0 < _n; n0++) {
                for (size_t i0 = 0; i0 < _i; i0++) {
                    for (size_t j0 = 0; j0 < _j; j0++) {
                        for (size_t e0 = 0; e0 < _e; e0++) {
                            for (size_t f0 = 0; f0 < _f; f0++) {
                                W0(m0, n0, i0, j0) += 0.25 * t_oovv(i0, j0, e0, f0) * g_oovv(m0, n0, e0, f0);
                            }
                        }
                    }
                }
            }
        }

        for (size_t m0 = 0; m0 < _m; m0++) {
            for (size_t n0 = 0; n0 < _n; n0++) {
                for (size_t i0 = 0; i0 < _i; i0++) {
                    for (size_t j0 = 0; j0 < _j; j0++) {
                        REQUIRE_THAT(Wmnij(m0, n0, i0, j0), Catch::Matchers::WithinAbs(W0(m0, n0, i0, j0), 0.001));
                    }
                }
            }
        }
    }
}

TEST_CASE("gemv jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    SECTION("check") {
        size_t _p = 7, _q = 7, _r = 7, _s = 7;

        Tensor g = create_random_tensor("g", _p, _q, _r, _s);
        Tensor D = create_random_tensor("d", _r, _s);

        Tensor F{"F", _p, _q};
        Tensor F0{"F0", _p, _q};

        zero(F);
        zero(F0);

        REQUIRE_NOTHROW(einsum(1.0, Indices{p, q}, &F0, 2.0, Indices{p, q, r, s}, g, Indices{r, s}, D));

        TensorView gv{g, Dim<2>{_p * _q, _r * _s}};
        TensorView dv{D, Dim<1>{_r * _s}};
        TensorView Fv{F, Dim<1>{_p * _q}};

        linear_algebra::gemv<false>(2.0, gv, dv, 1.0, &Fv);

        // println(F0);
        // println(F);
    }
}

TEST_CASE("outer product jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    size_t _x{100}, _y{100};

    SECTION("1 * 1 -> 2") {
        Tensor A = create_random_tensor("A", _x);
        Tensor B = create_random_tensor("B", _y);
        Tensor C{"C", _x, _y};
        zero(C);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }
    }

    SECTION("2 * 1 -> 3") {
        Tensor A = create_random_tensor("A", 3, 3);
        Tensor B = create_random_tensor("B", 3);
        Tensor C{"C", 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(x, y, z), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{k}, B, Indices{i, j}, A));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        Tensor A = create_random_tensor("A", 3, 3);
        Tensor B = create_random_tensor("B", 3, 3);
        Tensor C{"C", 3, 3, 3, 3};

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, A, Indices{k, l}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(w, x) * B(y, z), 0.001));
                    }
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{k, l}, A, Indices{i, j}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(y, z) * B(w, x), 0.001));
                    }
                }
            }
        }
    }
}

TEST_CASE("element transform jobs", "[jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    SECTION("tensor") {
        Tensor A     = create_random_tensor("A", 32, 32, 32, 32);
        Tensor Acopy = A;

        element_transform(&A, [](double val) -> double { return 1.0 / val; });

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(1.0 / Acopy(w, x, y, z), 0.001));
                    }
                }
            }
        }
    }

    SECTION("smartptr tensor") {
        auto A = std::make_unique<Tensor<double, 4>>("A", 32, 32, 32, 32);

        element_transform(&A, [](double val) -> double { return 1.0 / val; });
    }
}

TEST_CASE("einsum element jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    const int _i{50}, _j{50};

    SECTION("1") {
        Tensor C  = Tensor{"C", _i, _j};
        Tensor C0 = Tensor{"C", _i, _j};

        Tensor B = create_random_tensor("B", _i, _j);
        Tensor A = create_random_tensor("A", _i, _j);

        element([](double const & /*Cval*/, double const &Aval, double const &Bval) -> double { return Aval * Bval; }, &C0, A, B);

        einsum(Indices{i, j}, &C, Indices{i, j}, A, Indices{i, j}, B);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(C0(w, x), 1.0e-5));
            }
        }
    }

    SECTION("2") {
        Tensor C  = create_random_tensor("C", _i, _j);
        Tensor C0 = C;
        Tensor testresult{"result", _i, _j};
        zero(testresult);

        Tensor A = create_random_tensor("A", _i, _j);

        element([](double const &Cval, double const &Aval) -> double { return Cval * Aval; }, &C, A);

        einsum(Indices{i, j}, &testresult, Indices{i, j}, C0, Indices{i, j}, A);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(testresult(w, x), 1.0e-5));
            }
        }
    }

    SECTION("3") {
        Tensor parentC  = create_random_tensor("parentC", _i, _i, _i, _j);
        Tensor parentC0 = parentC;
        Tensor parentA  = create_random_tensor("parentA", _i, _i, _i, _j);

        auto   C  = parentC(3, All, All, 4);
        auto   C0 = parentC0(3, All, All, 4);
        Tensor testresult{"result", _i, _j};

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                testresult(w, x) = C(w, x);
            }
        }

        auto A = parentA(1, 2, All, All);

        element([](double const &Cval, double const &Aval) -> double { return Cval * Aval; }, &C, A);

        einsum(Indices{i, j}, &testresult, Indices{i, j}, C0, Indices{i, j}, A);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(testresult(w, x), 1.0e-5));
            }
        }
    }
}

TEST_CASE("F12 - V term jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{1}, ncabs{4}, nobs{2};
    int nall{nobs + ncabs};

    auto F = create_incremented_tensor("F", nall, nall, nall, nall);
    auto G = create_incremented_tensor("G", nall, nall, nall, nall);

    TensorView F_ooco{F, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    TensorView F_oooc{F, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    TensorView F_oopq{F, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};
    TensorView G_ooco{G, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    TensorView G_oooc{G, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    TensorView G_oopq{G, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};

    Tensor ijkl_1 = Tensor{"Einsum Temp 1", nocc, nocc, nocc, nocc};
    Tensor ijkl_2 = Tensor{"Einsum Temp 2", nocc, nocc, nocc, nocc};
    Tensor ijkl_3 = Tensor{"Einsum Temp 3", nocc, nocc, nocc, nocc};

    ijkl_1.set_all(0.0);
    ijkl_2.set_all(0.0);
    ijkl_3.set_all(0.0);

    Tensor result  = Tensor{"Result", nocc, nocc, nocc, nocc};
    Tensor result2 = Tensor{"Result2", nocc, nocc, nocc, nocc};

    // println(F);
    // println(G);

    einsum(Indices{i, j, k, l}, &ijkl_1, Indices{i, j, p, n}, G_ooco, Indices{k, l, p, n}, F_ooco);
    einsum(Indices{i, j, k, l}, &ijkl_2, Indices{i, j, m, q}, G_oooc, Indices{k, l, m, q}, F_oooc);
    einsum(Indices{i, j, k, l}, &ijkl_3, Indices{i, j, p, q}, G_oopq, Indices{k, l, p, q}, F_oopq);

    result.set_all(0.0);
    result2.set_all(0.0);
    timer::push("raw for loops");
    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    for (size_t _p = 0; _p < ncabs; _p++) {
                        for (size_t _n = 0; _n < nocc; _n++) {
                            // println("A({}, {}, {}, {}) = {}", _i, _j, _p, _n, G_ooco(_i, _j, _p, _n));
                            // println("B({}, {}, {}, {}) = {}", _k, _l, _p, _n, F_ooco(_k, _l, _p, _n));

                            result(_i, _j, _k, _l) += G(_i, _j, nobs + _p, _n) * F(_k, _l, nobs + _p, _n);
                            result2(_i, _j, _k, _l) += G_ooco(_i, _j, _p, _n) * F_ooco(_k, _l, _p, _n);
                        }
                    }
                }
            }
        }
    }
    timer::pop();

    // println(result);
    // println(ijkl_1);

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(result2(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(ijkl_1(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }
}

TEST_CASE("B_tilde jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{5}, ncabs{10}, nobs{10};
    assert(nobs > nocc); // sanity check
    int nall{nobs + ncabs}, nvir{nobs - nocc};

    Tensor CD{"CD", nocc, nocc, nvir, nvir};
    Tensor CD0{"CD0", nocc, nocc, nvir, nvir};
    zero(CD);
    zero(CD0);
    auto C    = create_random_tensor("C", nocc, nocc, nvir, nvir);
    auto D    = create_random_tensor("D", nocc, nocc, nvir, nvir);
    auto D_ij = D(2, 2, All, All);

    einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, C, Indices{a, b}, D_ij);

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    CD0(_k, _l, _a, _b) = C(_k, _l, _a, _b) * D(2, 2, _a, _b);
                }
            }
        }
    }

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    REQUIRE_THAT(CD(_k, _l, _a, _b), Catch::Matchers::WithinAbs(CD0(_k, _l, _a, _b), 0.000001));
                }
            }
        }
    }
}

TEST_CASE("Khatri-Rao jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    const int _I{8}, _M{4}, _r{16};

    SECTION("einsum") {

        auto KR  = Tensor{"KR", _I, _M, _r};
        auto KR0 = Tensor{"KR0", _I, _M, _r};

        auto T = create_random_tensor("T", _I, _r);
        auto U = create_random_tensor("U", _M, _r);

        einsum(Indices{I, M, r}, &KR, Indices{I, r}, T, Indices{M, r}, U);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    REQUIRE_THAT(KR(x, y, z), Catch::Matchers::WithinAbs(KR0(x, y, z), 0.000001));
                }
            }
        }
    }

    SECTION("special function") {
        auto KR0 = Tensor{"KR0", _I, _M, _r};

        auto T = create_random_tensor("T", _I, _r);
        auto U = create_random_tensor("U", _M, _r);

        auto KR = khatri_rao(Indices{I, r}, T, Indices{M, r}, U);
        // println(result);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        auto KR0_view = TensorView{KR0, Dim<2>{_I * _M, _r}};

        for (int x = 0; x < _I * _M; x++) {
            for (int z = 0; z < _r; z++) {
                REQUIRE_THAT(KR(x, z), Catch::Matchers::WithinAbs(KR0_view(x, z), 0.000001));
            }
        }
    }
}

TEST_CASE("andy jobs", "[unfinished-jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    size_t proj_rank_{10}, nocc_{5}, nvirt_{28}, naux_{3}, u_rank_{4};

    SECTION("1") {
        auto y_iW_        = create_random_tensor("y_iW", nocc_, proj_rank_);
        auto y_aW_        = create_random_tensor("y_aW", nvirt_, proj_rank_);
        auto ortho_temp_1 = create_tensor("ortho temp 1", nocc_, nvirt_, proj_rank_);

        zero(ortho_temp_1);
        einsum(0.0, Indices{index::i, index::a, index::W}, &ortho_temp_1, 1.0, Indices{index::i, index::W}, y_iW_,
               Indices{index::a, index::W}, y_aW_);
    }

    SECTION("2") {
        auto tau_         = create_random_tensor("tau", proj_rank_, proj_rank_);
        auto ortho_temp_1 = create_random_tensor("ortho temp 1", nocc_, nvirt_, proj_rank_);
        auto ortho_temp_2 = create_tensor("ortho temp 2", nocc_, nvirt_, proj_rank_);

        zero(ortho_temp_2);
        einsum(0.0, Indices{index::i, index::a, index::P}, &ortho_temp_2, 1.0, Indices{index::i, index::a, index::W}, ortho_temp_1,
               Indices{index::P, index::W}, tau_);
    }

    SECTION("3") {
        auto a = create_random_tensor("a", nvirt_, nvirt_);
        auto b = create_random_tensor("b", nvirt_, nvirt_);
        auto c = create_tensor("c", nvirt_, nvirt_);
        zero(c);

        einsum(0.0, Indices{index::p, index::q}, &c, -1.0, Indices{index::p, index::q}, a, Indices{index::p, index::q}, b);

        for (int x = 0; x < nvirt_; x++) {
            for (int y = 0; y < nvirt_; y++) {
                REQUIRE_THAT(c(x, y), Catch::Matchers::WithinRel(-a(x, y) * b(x, y)));
            }
        }
    }

    SECTION("4") {
        auto A  = create_random_tensor("a", proj_rank_, nocc_, nvirt_);
        auto B  = create_random_tensor("b", nocc_, nvirt_, proj_rank_);
        auto c  = create_tensor("c", proj_rank_, proj_rank_);
        auto c0 = create_tensor("c0", proj_rank_, proj_rank_);

        zero(c);
        einsum(Indices{index::Q, index::X}, &c, Indices{index::Q, index::i, index::a}, A, Indices{index::i, index::a, index::X}, B);

        zero(c0);
        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t X = 0; X < proj_rank_; X++) {
                for (size_t i = 0; i < nocc_; i++) {
                    for (size_t a = 0; a < nvirt_; a++) {
                        c0(Q, X) += A(Q, i, a) * B(i, a, X);
                    }
                }
            }
        }

        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t X = 0; X < proj_rank_; X++) {
                REQUIRE_THAT(c(Q, X), Catch::Matchers::WithinRel(c0(Q, X), 0.00001));
            }
        }
    }

    SECTION("5") {
        auto F_TEMP = create_random_tensor("F_TEMP", proj_rank_, proj_rank_, proj_rank_);
        auto y_aW   = create_random_tensor("y_aW", nvirt_, proj_rank_);
        auto F_BAR  = create_tensor("F_BAR", proj_rank_, nvirt_, proj_rank_);
        auto F_BAR0 = create_tensor("F_BAR", proj_rank_, nvirt_, proj_rank_);

        zero(F_BAR);
        einsum(Indices{index::Q, index::a, index::X}, &F_BAR, Indices{index::Q, index::Y, index::X}, F_TEMP, Indices{index::a, index::Y},
               y_aW);

        zero(F_BAR0);
        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t a = 0; a < nvirt_; a++) {
                for (size_t X = 0; X < proj_rank_; X++) {
                    for (size_t Y = 0; Y < proj_rank_; Y++) {
                        F_BAR0(Q, a, X) += F_TEMP(Q, Y, X) * y_aW(a, Y);
                    }
                }
            }
        }

        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t a = 0; a < nvirt_; a++) {
                for (size_t X = 0; X < proj_rank_; X++) {
                    REQUIRE_THAT(F_BAR(Q, a, X), Catch::Matchers::WithinRel(F_BAR0(Q, a, X), 0.00001));
                }
            }
        }
    }

    SECTION("6") {
        auto A = create_random_tensor("A", 84);
        auto C = create_tensor("C", 84, 84);
        zero(C);

        einsum(Indices{index::a, index::b}, &C, Indices{index::a}, A, Indices{index::b}, A);

        for (size_t a = 0; a < 84; a++) {
            for (size_t b = 0; b < 84; b++) {
                REQUIRE_THAT(C(a, b), Catch::Matchers::WithinRel(A(a) * A(b), 0.00001));
            }
        }
    }

    SECTION("7") {
        auto A = create_tensor("A", 9);
        A(0)   = 0.26052754;
        A(1)   = 0.20708203;
        A(2)   = 0.18034861;
        A(3)   = 0.18034861;
        A(4)   = 0.10959806;
        A(5)   = 0.10285149;
        A(6)   = 0.10285149;
        A(7)   = 0.10164104;
        A(8)   = 0.06130642;
        auto C = create_tensor("C", 9, 9);
        zero(C);

        einsum(Indices{index::a, index::b}, &C, Indices{index::a}, A, Indices{index::b}, A);

        for (size_t a = 0; a < 9; a++) {
            for (size_t b = 0; b < 9; b++) {
                REQUIRE_THAT(C(a, b), Catch::Matchers::WithinRel(A(a) * A(b), 0.00001));
            }
        }
    }

    SECTION("8") {
        auto C_TILDE = create_random_tensor("C_TILDE", naux_, nvirt_, u_rank_);
        auto B_QY    = create_random_tensor("B_QY", naux_, u_rank_);

        auto D_TILDE = create_tensor("D_TILDE", nvirt_, u_rank_);
        zero(D_TILDE);

        einsum(0.0, Indices{index::a, index::X}, &D_TILDE, 1.0, Indices{index::Q, index::a, index::X}, C_TILDE, Indices{index::Q, index::X},
               B_QY);
    }

    SECTION("9") {
        auto Qov  = create_random_tensor("Qov", naux_, nocc_, nvirt_);
        auto ia_X = create_random_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto N_QX = create_tensor("N_QX", naux_, u_rank_);
        zero(N_QX);

        einsum(Indices{index::Q, index::X}, &N_QX, Indices{index::Q, index::i, index::a}, Qov, Indices{index::i, index::a, index::X}, ia_X);
    }

    SECTION("10") {
        auto t_ia = create_random_tensor("t_ia", nocc_, nvirt_);
        auto ia_X = create_random_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto M_X = create_tensor("M_X", u_rank_);
        zero(M_X);

        einsum(Indices{index::X}, &M_X, Indices{index::i, index::a, index::X}, ia_X, Indices{index::i, index::a}, t_ia);
    }

    SECTION("11") {
        auto B_Qmo = create_random_tensor("Q", naux_, nocc_ + nvirt_, nocc_ + nvirt_);
        // println(B_Qmo);
        auto Qov = B_Qmo(All, Range{0, nocc_}, Range{nocc_, nocc_ + nvirt_});

        // println(Qov, {.full_output = false});

        auto ia_X = create_random_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto N_QX = create_tensor("N_QX", naux_, u_rank_);
        zero(N_QX);

        einsum(Indices{index::Q, index::X}, &N_QX, Indices{index::Q, index::i, index::a}, Qov, Indices{index::i, index::a, index::X}, ia_X);
    }
}


TEST_CASE("Complicated Jobs", "[jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::jobs;

    // Start the job manager.
    ThreadPool::init(8);
    JobManager::get_singleton().start_manager();
    
    // Create several tensor resources.
    std::shared_ptr<jobs::Resource<Tensor<double, 2>>> A_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("A", 3, 3),
        B_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("B", 3, 3),
	C_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("C", 3, 3),
	D_res = std::make_shared<jobs::Resource<Tensor<double, 2>>>("D", 3, 3);

    // Get locks for writing A and B.
    auto A_lock1 = A_res->write_promise();
    auto B_lock1 = B_res->write_promise();

    // Queue up a second layer to test resolution.
    auto A_lock2 = A_res->write_promise();
    auto B_lock2 = B_res->write_promise();

    // Make sure the second locks don't resolve.
    REQUIRE_THROWS_AS(A_lock2->wait(std::chrono::milliseconds(10)), jobs::timeout);
    REQUIRE_THROWS_AS(B_lock2->wait(std::chrono::milliseconds(10)), jobs::timeout);

    // Resolve the first locks for zeroing the tensors.
    auto A = A_lock1->get();
    auto B = B_lock1->get();

    // Zero the matrices.
    A->zero();
    B->zero();

    A_lock1->release();
    B_lock1->release();

    // Queue up jobs for the other tensors.
    jobs::einsum(Indices{index::i, index::j}, C_res, Indices{index::i, index::l}, A_res, Indices{index::l, index::j}, B_res);
    jobs::einsum(Indices{index::i, index::j}, D_res, Indices{index::i, index::l}, B_res, Indices{index::l, index::j}, A_res);
    jobs::einsum(Indices{index::i, index::j}, A_res, Indices{index::i, index::j}, B_res, Indices{index::j, index::i}, B_res);

    // Set A and B.
    A = A_lock2->get();
    B = B_lock2->get();

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
	    (*A)(i, j) = i * 3 + j + 1;
	    (*B)(i, j) = i * 3 + j + 1;
	}
    }

    A_lock2->release();
    B_lock2->release();

    auto A_lock = A_res->read_promise();
    auto C_lock = C_res->read_promise();
    auto D_lock = D_res->read_promise();

    A = A_lock->get();
    auto C = C_lock->get();
    auto D = D_lock->get();

    REQUIRE((*A)(0, 0) == 1);
    REQUIRE((*A)(0, 1) == 8);
    REQUIRE((*A)(0, 2) == 21);
    REQUIRE((*A)(1, 0) == 8);
    REQUIRE((*A)(1, 1) == 25);
    REQUIRE((*A)(1, 2) == 48);
    REQUIRE((*A)(2, 0) == 21);
    REQUIRE((*A)(2, 1) == 48);
    REQUIRE((*A)(2, 2) == 81);

    REQUIRE((*C)(0, 0) == 30);
    REQUIRE((*C)(0, 1) == 36);
    REQUIRE((*C)(0, 2) == 42);
    REQUIRE((*C)(1, 0) == 66);
    REQUIRE((*C)(1, 1) == 81);
    REQUIRE((*C)(1, 2) == 96);
    REQUIRE((*C)(2, 0) == 102);
    REQUIRE((*C)(2, 1) == 126);
    REQUIRE((*C)(2, 2) == 150);

    REQUIRE((*D)(0, 0) == 30);
    REQUIRE((*D)(0, 1) == 36);
    REQUIRE((*D)(0, 2) == 42);
    REQUIRE((*D)(1, 0) == 66);
    REQUIRE((*D)(1, 1) == 81);
    REQUIRE((*D)(1, 2) == 96);
    REQUIRE((*D)(2, 0) == 102);
    REQUIRE((*D)(2, 1) == 126);
    REQUIRE((*D)(2, 2) == 150);

    A_lock->release();
    C_lock->release();
    D_lock->release();

    JobManager::get_singleton().wait_on_jobs();
    ThreadPool::get_singleton().wait_on_running();
    JobManager::get_singleton().stop_manager();
    ThreadPool::destroy();
    JobManager::destroy();

}
