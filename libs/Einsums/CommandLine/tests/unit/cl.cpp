//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums::cl;

struct CLITestFixture {
    CLITestFixture() { Registry::instance().clear_for_tests(); }
    ~CLITestFixture() { Registry::instance().clear_for_tests(); }
};

static std::vector<std::string> to_args(std::initializer_list<char const *> il) {
    std::vector<std::string> v;
    v.reserve(il.size());
    for (auto *s : il)
        v.emplace_back(s);
    return v;
}

TEST_CASE("Defaults and explicit override", "[opt][defaults]") {
    CLITestFixture _;

    OptionCategory Cat{"T1"};
    int            threads_bound = 0;
    Opt<int>       Threads{"t1-threads", {'T'}, "threads", Default(4), Cat, RangeBetween(1, 256), ValueName("N")};
    Threads.OnSet([&](int v) { threads_bound = v; });

    SECTION("No args -> default 4") {
        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Threads.get() == 4);
        REQUIRE(threads_bound == 4);
    }

    SECTION("Explicit CLI override -> 8") {
        auto args = to_args({"prog", "--t1-threads", "8"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Threads.get() == 8);
        REQUIRE(threads_bound == 8);
    }
}

TEST_CASE("Implicit value when present without argument", "[opt][implicit]") {
    CLITestFixture _;

    OptionCategory Cat{"T2"};
    Opt<int>       Level{"t2-level", {'l'}, "level", Cat, ImplicitValue(7), ValueName("N")};

    SECTION("Appears without value -> implicit 7") {
        auto args = to_args({"prog", "--t2-level"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Level.get() == 7);
    }

    SECTION("Appears with value -> explicit wins (9)") {
        auto args = to_args({"prog", "--t2-level=9"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Level.get() == 9);
    }
}

TEST_CASE("Flag default and implicit override", "[flag]") {
    OptionCategory Cat{"T3"};
    bool           verbose_bound = false;
    Flag           Verbose{"t3-verbose", {'v'}, "verbose logging", Cat, Default(false), ImplicitValue(true)};
    Verbose.OnSet([&](bool on) { verbose_bound = on; });

    SECTION("Default false") {
        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE_FALSE(Verbose.get());
        REQUIRE_FALSE(verbose_bound);
    }

    SECTION("Presence -> true") {
        auto args = to_args({"prog", "-v"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(Verbose.get());
        REQUIRE(verbose_bound);
    }

    SECTION("Explicit false via value") {
        auto args = to_args({"prog", "--t3-verbose=false"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE_FALSE(Verbose.get());
    }
}

TEST_CASE("Bundled short options and attached value", "[short][bundle]") {
    CLITestFixture _;

    OptionCategory Cat{"T4"};
    Flag           A{"t4-a", {'a'}, "flag A", Cat};
    Flag           B{"t4-b", {'b'}, "flag B", Cat};
    Opt<int>       O{"t4-o", {'o'}, "opt O", Cat, ValueName("N")};

    SECTION("-ab sets both; -o12 attaches value") {
        auto args = to_args({"prog", "-ab", "-o12"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(A.get());
        REQUIRE(B.get());
        REQUIRE(O.get() == 12);
    }

    SECTION("-o 34 with space") {
        auto args = to_args({"prog", "-o", "34"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(O.get() == 34);
    }
}

TEST_CASE("Positional list captures multiple tokens", "[positional][list]") {
    CLITestFixture _;

    OptionCategory    Cat{"T5"};
    List<std::string> Inputs{"t5-inputs", Positional{}, "inputs"};

    auto args = to_args({"prog", "a.txt", "b.txt", "c.txt"});
    auto pr   = parse(args);
    REQUIRE(pr.ok);
    auto const &vals = Inputs.values();
    REQUIRE(vals.size() == 3);
    REQUIRE(vals[0] == "a.txt");
    REQUIRE(vals[1] == "b.txt");
    REQUIRE(vals[2] == "c.txt");
}

#if 0
TEST_CASE("Enum option maps strings to enum values", "[enum]") {
    CLITestFixture _;

    static OptionCategory Cat{"T6"};
    enum struct Mode { Fast, Accurate };
    static OptEnum<Mode> M{"t6-mode", {'m'}, Mode::Fast, {{"fast", Mode::Fast}, {"accurate", Mode::Accurate}}, "mode", Cat};

    SECTION("Default is Fast") {
        auto args = to_args({"prog"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(M.to_string() == "fast");
    }

    SECTION("Set to accurate") {
        auto args = to_args({"prog", "--t6-mode", "accurate"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(M.to_string() == "accurate");
    }

    SECTION("Invalid value errors") {
        auto args = to_args({"prog", "--t6-mode", "banana"});
        auto pr   = parse(args);
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 1);
    }
}
#endif

TEST_CASE("Range validation enforces bounds", "[range]") {
    CLITestFixture _;

    OptionCategory Cat{"T7"};
    Opt<int>       R{"t7-ranged", {'r'}, "ranged", Default(10), Cat, RangeBetween(5, 15)};

    SECTION("In range ok") {
        auto args = to_args({"prog", "--t7-ranged=12"});
        auto pr   = parse(args);
        REQUIRE(pr.ok);
        REQUIRE(R.get() == 12);
    }

    SECTION("Out of range produces error") {
        auto args = to_args({"prog", "--t7-ranged=100"});
        auto pr   = parse(args);
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 1);
    }
}

TEST_CASE("Config precedence: defaults < config < CLI", "[config][precedence]") {
    CLITestFixture _;

    OptionCategory Cat{"T8"};
    int            threads_cfg = 0;
    Opt<int>       T{"t8-threads", {'t'}, "threads", Default(2), Cat, Location<int>(threads_cfg)};

    std::map<std::string, std::string, std::less<>> cfg;
    cfg["t8-threads"] = "6";

    SECTION("Config overrides default") {
        auto                     args = to_args({"prog"});
        std::vector<std::string> unknown;
        auto                     pr = parse_internal(args, "prog", "1.0", &cfg, &unknown);
        REQUIRE(pr.ok);
        REQUIRE(T.get() == 6);
        REQUIRE(threads_cfg == 6);
    }

    SECTION("CLI overrides config") {
        auto                     args = to_args({"prog", "--t8-threads=9"});
        std::vector<std::string> unknown;
        auto                     pr = parse_internal(args, "prog", "1.0", &cfg, &unknown);
        REQUIRE(pr.ok);
        REQUIRE(T.get() == 9);
        REQUIRE(threads_cfg == 9);
    }
}

TEST_CASE("Unknown args collection (including after --)", "[unknown]") {
    CLITestFixture _;

    OptionCategory Cat{"T9"};
    Flag           K{"t9-known", {'k'}, "known", Cat};

    auto                     args = to_args({"prog", "--nope", "-z", "--", "pos1", "--still", "-x"});
    std::vector<std::string> unknown;
    auto                     pr = parse_internal(args, "prog", "1.0", nullptr, &unknown);
    REQUIRE(pr.ok);

    REQUIRE_FALSE(K.get()); // known flag not present

    REQUIRE(unknown.size() == 5);
    REQUIRE(unknown[0] == "--nope");
    REQUIRE(unknown[1] == "-z");
    REQUIRE(unknown[2] == "pos1");
    REQUIRE(unknown[3] == "--still");
    REQUIRE(unknown[4] == "-x");
}

TEST_CASE("Builtins: --help and --version exit 0", "[builtins]") {
    CLITestFixture _;

    SECTION("--help exits 0") {
        auto args = to_args({"prog", "--help"});
        auto pr   = parse(args, "prog", "9.9");
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 0);
    }
    SECTION("--version exits 0") {
        auto args = to_args({"prog", "--version"});
        auto pr   = parse(args, "prog", "9.9");
        REQUIRE_FALSE(pr.ok);
        REQUIRE(pr.exit_code == 0);
    }
}