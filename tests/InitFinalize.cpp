#include <einsums.hpp>
#include <catch2/catch_all.hpp>
#include <sys/stat.h>
#include <unistd.h>

TEST_CASE("Initialize Finalize") {
    einsums::initialize();

    struct stat buffer;

    // Clear timing files.
    int result = stat("timings.txt", &buffer);

    if(result == 0) {
        remove("timings.txt");
    }

    result = stat("timings_2.txt", &buffer);

    if(result == 0) {
        remove("timings_2.txt");
    }

    SECTION("No print") {
        einsums::finalize(false);
    }

    SECTION("Print to standard file") {
        einsums::finalize(true);
    }

    SECTION("Print to other file") {
        einsums::finalize("timings_2.txt");
    }

    einsums::initialize();
}