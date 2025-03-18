#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include "Einsums/BufferAllocator/ModuleVars.hpp"

#include <Einsums/Testing.hpp>

TEST_CASE("Allocations") {

    using namespace einsums;

    auto &config = GlobalConfigMap::get_singleton();

    auto hold_str = config.get_string("buffer-size");

    {
        auto lock = std::lock_guard(config);

        config.get_string_map()->get_value()["buffer-size"] = "100"; // Set to some small number of bytes.
    }

    // First, check for the new value of the buffer size.
    auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();
    REQUIRE(vars.get_max_size() == 100);

    // Allocate stuff.
    std::vector<int, BufferAllocator<int>> vec1;

    REQUIRE_NOTHROW(vec1.resize(1));
    REQUIRE(vec1.size() == 1);
    REQUIRE(vars.get_available() == 96);
    REQUIRE_NOTHROW(vec1.at(0) = 0);
    REQUIRE(vec1[0] == 0);

    REQUIRE_NOTHROW(vec1.resize(20));
    REQUIRE(vec1.size() == 20);
    REQUIRE(vars.get_available() == 20);
    REQUIRE_NOTHROW(vec1.at(19) = 0);
    REQUIRE(vec1[19] == 0);

    vec1.clear();
    vec1.shrink_to_fit();

    REQUIRE(vars.get_available() == 100);

    REQUIRE_THROWS(vec1.resize(100 / sizeof(int) + 1));
    
}