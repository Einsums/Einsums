#include <einsums.hpp>
#include <catch2/catch_all.hpp>
#include <sys/stat.h>
#include <error.h>
#include <errno.h>
//#include <unistd.h>

TEST_CASE("Initialize Finalize") {
    einsums::initialize();

    struct stat buffer;

    // Clear timing files.
    int result = stat("timings.txt", &buffer);

    if(result == 0) {
        remove("timings.txt");
    }

    errno = 0;

    SECTION("No print") {
        einsums::finalize(false);

        REQUIRE(stat("timings.txt", &buffer) != 0);
    }

    SECTION("Print to standard output") {
        einsums::finalize(true);

        REQUIRE(stat("timings.txt", &buffer) != 0);
    }

    SECTION("Print to other file") {
        einsums::finalize("timings.txt");
        
        result = stat("timings.txt", &buffer);

        if(result != 0) {
            perror("Could not stat file!");
        }

        REQUIRE(result == 0);
    }

    einsums::initialize();
}