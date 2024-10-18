#include <catch2/catch_all.hpp>
#include <einsums.hpp>
#include <errno.h>
#include <error.h>
#include <fstream>
#include <sys/stat.h>
// #include <unistd.h>

TEST_CASE("Initialize Finalize") {
    einsums::initialize();

    struct stat buffer;

    // Clear timing files.
    int result = stat("timings.txt", &buffer);

    if (result == 0) {
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

    SECTION("Print to other file - char *") {
        einsums::finalize("timings.txt");

        result = stat("timings.txt", &buffer);

        if (result != 0) {
            perror("Could not stat file!");
        }

        REQUIRE(result == 0);
    }

    SECTION("Print to other file - std::string") {
        einsums::finalize(std::string{"timings.txt"});

        result = stat("timings.txt", &buffer);

        if (result != 0) {
            perror("Could not stat file!");
        }

        REQUIRE(result == 0);
    }

    SECTION("Print to ostream") {
        auto stream = std::ofstream("timings.txt");
        einsums::finalize(stream);

        stream.close();

        result = stat("timings.txt", &buffer);

        if (result != 0) {
            perror("Could not stat file!");
        }

        REQUIRE(result == 0);
    }

    SECTION("Print to file buffer") {
        auto fp = std::fopen("timings.txt", "w");
        einsums::finalize(fp);

        std::fclose(fp);
        
        result = stat("timings.txt", &buffer);

        if (result != 0) {
            perror("Could not stat file!");
        }

        REQUIRE(result == 0);
    }

    einsums::initialize();
}