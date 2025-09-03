//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

// #include <Einsums/Config.hpp>

// Lightweight, header-only command-line interface inspired by LLVM's cl::opt
// Additions in this revision:
//  - Subcommands (LLVM-style): declare `Subcommand Tools{"tools"};` and any
//    options declared after will belong to that subcommand. Root options belong
//    to the implicit `Root()` subcommand.
//  - Location<T>: bind options directly to external storage (variable reference).
//  - Config precedence: defaults < config file < CLI. Use `parse_with_config(...)`.
//  - Flat JSON (and key=value) config reader with no external deps.
//
// Design goals:
//  - No external deps beyond fmtlib (used for printing)
//  - Header-only, C++20, no RTTI or exceptions required (but works with them)
//  - Registration-based: define options as globals or in static scope
//  - Supports: --long, -s, positional, lists (multi-occurrence), aliasing,
//              categories, hidden, required, default values, range checks,
//              enum adapters, custom parsers, help/version banners,
//              subcommands, Location<T>, config file precedence
//
// Usage (typical):
//   using namespace ein::cli;
//   static OptionCategory Cat{"General Options"};
//   static Flag        OptVerbose{"verbose", {'v'}, "Enable verbose output", Cat};
//   static Opt<int>    OptI{"iters", {'i'}, 10, "Number of iterations", Cat,
//                            Range{1, 1'000'000}};
//   enum class Mode { Fast, Accurate };
//   static OptEnum<Mode> OptMode{"mode", {}, Mode::Fast,
//       { {"fast", Mode::Fast}, {"accurate", Mode::Accurate} },
//       "Execution mode", Cat};
//   static List<std::string> PosFiles{"files", Positional{}, "Input files"};
//   int main(int argc, char** argv) {
//     ParseResult pr = parse_with_config(argc, argv, /*programName*/nullptr,
//                           /*version*/"1.0.0", /*configPath*/"settings.json");
//     if (!pr.ok) return pr.exit_code; // help/--version or error already printed
//     fmt::print("verbose={} iters={} mode={} files={}",
//                OptVerbose.get(), OptI.get(), OptMode.to_string().c_str(),
//                PosFiles.values().size());
//   }

#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace einsums::cl {

// -------------------------- Small utilities ------------------------------- //

struct StringRef {
    std::string_view s;
    constexpr StringRef() = default;
    constexpr StringRef(char const *c) : s(c) {}
    constexpr StringRef(std::string_view v) : s(v) {}
    constexpr operator std::string_view() const { return s; }
};

struct Range {
    long long min_v = (std::numeric_limits<long long>::min)();
    long long max_v = (std::numeric_limits<long long>::max)();
};

// Result of parse() top-level; ok==false and exit_code!=0 => error was printed.
// ok==false and exit_code==0 => informational (help/version) printed; exit.
struct ParseResult {
    bool ok        = true;
    int  exit_code = 0;
};

// Forward decls
struct OptionBase;
struct OptionCategory;
struct Subcommand;

// -------------------------- Registry ------------------------------------- //

struct Registry {
    std::vector<OptionBase *>     options;
    std::vector<OptionCategory *> categories;
    std::vector<Subcommand *>     subcommands;
    bool                          show_hidden_by_default = false;

    static Registry &instance();
    void             add_option(OptionBase *o) { options.push_back(o); }
    void             add_category(OptionCategory *c) { categories.push_back(c); }
    void             add_subcommand(Subcommand *s) { subcommands.push_back(s); }
};

inline Registry &Registry::instance() {
    static Registry R;
    return R;
}

// -------------------------- Categories ----------------------------------- //

struct OptionCategory {
    std::string name;
    explicit OptionCategory(StringRef n) : name(n.s) { Registry::instance().add_category(this); }
};

// -------------------------- Subcommands ---------------------------------- //

struct Subcommand {
    std::string name; // e.g. "opt", "bench"
    std::string help;
    bool        is_root = false; // implicit root subcommand
    explicit Subcommand(StringRef n, StringRef h = {}) : name(n.s), help(h.s) {
        Registry::instance().add_subcommand(this);
        current() = this; // options declared after attach to this subcommand
    }
    static Subcommand &Root() noexcept {
        static Subcommand R{"<root>", ""};
        R.is_root = true;
        return R;
    }
    // Global current-subcommand pointer for registration-time association
    static Subcommand *&current() noexcept {
        static Subcommand *cur = &Root();
        return cur;
    }
};

// RAII guard to temporarily register options under a subcommand when used in a scope
struct SubcommandScope {
    Subcommand *prev;
    SubcommandScope(Subcommand &sc) : prev(Subcommand::current()) { Subcommand::current() = &sc; }
    ~SubcommandScope() { Subcommand::current() = prev; }
};

// -------------------------- Parser concept -------------------------------- //

// Default value parsers; users can specialize/overload parse_value for custom T

template <typename T>
static bool parse_value(std::string_view, T &out, std::string &err);

inline bool parse_value(std::string_view sv, std::string &out, std::string &) {
    out.assign(sv.begin(), sv.end());
    return true;
}

inline bool parse_value(std::string_view sv, bool &out, std::string &err) {
    if (sv == "1" || sv == "true" || sv == "on" || sv == "yes") {
        out = true;
        return true;
    }
    if (sv == "0" || sv == "false" || sv == "off" || sv == "no") {
        out = false;
        return true;
    }
    err = fmt::format("invalid boolean '{}', expected true/false", sv);
    return false;
}

inline bool parse_value(std::string_view sv, int &out, std::string &err) {
    auto *b = sv.data();
    auto *e = b + sv.size();
    int   v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    err = fmt::format("invalid integer '{}'", sv);
    return false;
}

inline bool parse_value(std::string_view sv, long long &out, std::string &err) {
    auto     *b = sv.data();
    auto     *e = b + sv.size();
    long long v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    err = fmt::format("invalid integer '{}'", sv);
    return false;
}

inline bool parse_value(std::string_view sv, double &out, std::string &err) {
    try {
        size_t      idx = 0;
        std::string tmp(sv);
        double      v = std::stod(tmp, &idx);
        if (idx == tmp.size()) {
            out = v;
            return true;
        }
    } catch (...) {
    }
    err = fmt::format("invalid real '{}'", sv);
    return false;
}

// -------------------------- Option Base ---------------------------------- //

enum struct Visibility : uint8_t { Normal, Hidden };

enum struct Occurrence : uint8_t { Optional, Required, ZeroOrMore, OneOrMore };

enum struct ValueExpected : uint8_t { ValueDisallowed, ValueOptional, ValueRequired };

struct Positional {};

struct OptionBase {
    std::string       long_name;   // e.g. "verbose"
    std::vector<char> short_names; // e.g. {'v'}
    std::string       help;
    OptionCategory   *category       = nullptr;
    Visibility        visibility     = Visibility::Normal;
    Occurrence        occurrence     = Occurrence::Optional;
    ValueExpected     value_expected = ValueExpected::ValueOptional;
    bool              is_positional  = false;               // true => consumes positional tokens
    bool              seen_cli       = false;               // set if seen on CLI
    bool              seen_config    = false;               // set if set from config file
    int               occurrences    = 0;                   // number of times provided (CLI only)
    Subcommand       *subcommand     = &Subcommand::Root(); // owning subcommand

    // Hooks
    std::function<void()> on_seen; // after successfully parsed (CLI)

    OptionBase(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, OptionCategory *cat)
        : long_name(longName.s), short_names(shorts), help(helpText.s), category(cat), subcommand(Subcommand::current()) {
        Registry::instance().add_option(this);
    }

    OptionBase(StringRef positional_name, Positional, StringRef helpText)
        : long_name(positional_name.s), help(helpText.s), is_positional(true) {
        subcommand = Subcommand::current();
        Registry::instance().add_option(this);
    }

    virtual ~OptionBase() = default;

    virtual bool parse_token(std::string_view key, std::optional<std::string_view> val, std::string &error,
                             bool from_config = false)                                           = 0; // return true on success
    virtual void print_help_line(std::string_view prog, size_t pad_long, size_t pad_short) const = 0;
    virtual void finalize_default() {} // invoked before parse to initialize display state
    virtual bool validate(std::string &error) const {
        (void)error;
        return true;
    }
    bool active_for(Subcommand *current) const { return subcommand == &Subcommand::Root() || subcommand == current; }
};

// -------------------------- Location<T> helper ---------------------------- //

template <typename T>
struct Location {
    T *ptr = nullptr;
    explicit Location(T &r) : ptr(&r) {}
};

// -------------------------- Flag (bool) ---------------------------------- //

struct Flag : OptionBase {
    bool  value = false;
    bool *bound = nullptr;

    explicit Flag(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, OptionCategory &cat,
                  Visibility vis = Visibility::Normal, Occurrence occ = Occurrence::Optional)
        : OptionBase(longName, shorts, helpText, &cat) {
        visibility     = vis;
        occurrence     = occ;
        value_expected = ValueExpected::ValueOptional;
    }

    explicit Flag(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, Visibility vis = Visibility::Normal,
                  Occurrence occ = Occurrence::Optional)
        : OptionBase(longName, shorts, helpText, nullptr) {
        visibility     = vis;
        occurrence     = occ;
        value_expected = ValueExpected::ValueOptional;
    }

    explicit Flag(StringRef longName, std::initializer_list<char> shorts, Location<bool> loc, StringRef helpText, OptionCategory &cat)
        : OptionBase(longName, shorts, helpText, &cat), bound(loc.ptr) {
        value_expected = ValueExpected::ValueOptional;
    }

    bool parse_token(std::string_view key, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        (void)key;
        bool tmp;
        if (!val.has_value()) {
            tmp = true;
        } else if (!parse_value(*val, tmp, error))
            return false;
        if (from_config) {
            value = tmp;
            if (bound)
                *bound = value;
            seen_config = true;
            return true;
        }
        value = tmp;
        if (bound)
            *bound = value;
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.pop_back(), shorts.pop_back();
        fmt::print("  {:<{}}  {:<{}}  {}\n", fmt::format("--{}", long_name), pad_long, shorts, pad_short, help);
    }

    bool get() const { return bound ? *bound : value; }
};

// -------------------------- Opt<T> --------------------------------------- //

template <typename T>
struct Opt : OptionBase {
    T                    value{};
    T                   *bound       = nullptr;
    bool                 has_default = false;
    std::optional<Range> range;
    std::optional<T>     implicit_value;       // if provided without value
    std::string          value_name = "value"; // for help

    Opt(StringRef longName, std::initializer_list<char> shorts, T defaultValue, StringRef helpText, OptionCategory &cat,
        Visibility vis = Visibility::Normal, Occurrence occ = Occurrence::Optional, ValueExpected ve = ValueExpected::ValueRequired,
        std::optional<Range> valid = std::nullopt)
        : OptionBase(longName, shorts, helpText, &cat), value(defaultValue) {
        visibility     = vis;
        occurrence     = occ;
        value_expected = ve;
        has_default    = true;
        range          = valid;
    }

    Opt(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, OptionCategory &cat,
        Visibility vis = Visibility::Normal, Occurrence occ = Occurrence::Optional, ValueExpected ve = ValueExpected::ValueRequired,
        std::optional<Range> valid = std::nullopt)
        : OptionBase(longName, shorts, helpText, &cat) {
        visibility     = vis;
        occurrence     = occ;
        value_expected = ve;
        range          = valid;
    }

    Opt(StringRef longName, std::initializer_list<char> shorts, Location<T> loc, StringRef helpText, OptionCategory &cat,
        ValueExpected ve = ValueExpected::ValueRequired)
        : OptionBase(longName, shorts, helpText, &cat), bound(loc.ptr) {
        value_expected = ve;
    }

    Opt &Implicit(T v) {
        implicit_value = v;
        return *this;
    }
    Opt &ValueName(StringRef n) {
        value_name = std::string(n.s);
        return *this;
    }

    template <typename U = T>
    bool assign_checked(T const &tmp, std::string &error) {
        if constexpr (std::is_arithmetic_v<U>) {
            if (range.has_value()) {
                long long vll = static_cast<long long>(tmp);
                if (vll < range->min_v || vll > range->max_v) {
                    error = fmt::format("value for '--{}' out of range [{}, {}]", long_name, range->min_v, range->max_v);
                    return false;
                }
            }
        }
        value = tmp;
        if (bound)
            *bound = value;
        return true;
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        if (!val.has_value()) {
            if (value_expected == ValueExpected::ValueRequired) {
                if (implicit_value.has_value()) {
                    if (!assign_checked(*implicit_value, error))
                        return false;
                } else {
                    error = fmt::format("option '--{}' requires a value", long_name);
                    return false;
                }
            }
            if (from_config) {
                seen_config = true;
                return true;
            }
            seen_cli = true;
            ++occurrences;
            return true;
        }
        T tmp{};
        if (!parse_value(*val, tmp, error))
            return false;
        if (!assign_checked(tmp, error))
            return false;
        if (from_config) {
            seen_config = true;
            return true;
        }
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.pop_back(), shorts.pop_back();
        std::string def;
        if (has_default && !bound)
            def = fmt::format(" (default: {})", value);
        fmt::print("  {:<{}}  {:<{}}  {}{}\n", fmt::format("--{} <{}>", long_name, value_name), pad_long, shorts, pad_short, help, def);
    }

    T const &get() const { return bound ? *bound : value; }
};

// -------------------------- List<T> -------------------------------------- //

template <typename T>
struct List : OptionBase {
    std::vector<T> vals;

    // Named list: e.g. --include a --include b  OR  --include=a,b
    List(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, OptionCategory &cat,
         Visibility vis = Visibility::Normal, Occurrence occ = Occurrence::Optional)
        : OptionBase(longName, shorts, helpText, &cat) {
        visibility     = vis;
        occurrence     = occ;
        value_expected = ValueExpected::ValueRequired;
    }

    // Positional list (captures remaining tokens)
    List(StringRef positional_name, Positional, StringRef helpText) : OptionBase(positional_name, Positional{}, helpText) {
        is_positional  = true;
        value_expected = ValueExpected::ValueRequired;
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        if (!val.has_value()) {
            error = fmt::format("option '--{}' requires a value", long_name);
            return false;
        }
        std::string_view s     = *val;
        size_t           start = 0;
        while (start <= s.size()) {
            size_t           comma = s.find(',', start);
            std::string_view item  = (comma == std::string_view::npos) ? s.substr(start) : s.substr(start, comma - start);
            if (!item.empty()) {
                T tmp{};
                if (!parse_value(item, tmp, error))
                    return false;
                vals.push_back(std::move(tmp));
            }
            if (comma == std::string_view::npos)
                break;
            start = comma + 1;
        }
        if (from_config) {
            seen_config = true;
            return true;
        }
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.pop_back(), shorts.pop_back();
        fmt::print("  {:<{}}  {:<{}}  {}\n", fmt::format("--{} <v1,v2,...>", long_name), pad_long, shorts, pad_short, help);
    }

    std::vector<T> const &values() const { return vals; }
};

// -------------------------- Enum adapter --------------------------------- //

template <typename Enum>
struct OptEnum : OptionBase {
    Enum                                     value{};
    Enum                                    *bound       = nullptr;
    bool                                     has_default = false;
    std::map<std::string, Enum, std::less<>> mapping; // name -> enum

    OptEnum(StringRef longName, std::initializer_list<char> shorts, Enum defaultValue,
            std::initializer_list<std::pair<std::string, Enum>> map, StringRef helpText, OptionCategory &cat,
            Visibility vis = Visibility::Normal, Occurrence occ = Occurrence::Optional)
        : OptionBase(longName, shorts, helpText, &cat), value(defaultValue), has_default(true), mapping(map) {
        (void)vis;
        (void)occ;
        value_expected = ValueExpected::ValueRequired;
    }

    OptEnum(StringRef longName, std::initializer_list<char> shorts, std::initializer_list<std::pair<std::string, Enum>> map,
            StringRef helpText, OptionCategory &cat)
        : OptionBase(longName, shorts, helpText, &cat), mapping(map) {
        value_expected = ValueExpected::ValueRequired;
    }

    OptEnum(StringRef longName, std::initializer_list<char> shorts, Location<Enum> loc,
            std::initializer_list<std::pair<std::string, Enum>> map, StringRef helpText, OptionCategory &cat)
        : OptionBase(longName, shorts, helpText, &cat), bound(loc.ptr), mapping(map) {
        value_expected = ValueExpected::ValueRequired;
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        if (!val.has_value()) {
            error = fmt::format("option '--{}' requires a value", long_name);
            return false;
        }
        auto it = mapping.find(std::string(*val));
        if (it == mapping.end()) {
            std::string keys;
            size_t      i = 0;
            for (auto &kv : mapping) {
                keys += kv.first;
                if (++i < mapping.size())
                    keys += ", ";
            }
            error = fmt::format("invalid value '{}' for '--{}' (choices: {})", *val, long_name, keys);
            return false;
        }
        if (bound)
            *bound = it->second;
        else
            value = it->second;
        if (from_config) {
            seen_config = true;
            return true;
        }
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.pop_back(), shorts.pop_back();
        std::string keys;
        size_t      i = 0;
        for (auto &kv : mapping) {
            keys += kv.first;
            if (++i < mapping.size())
                keys += "|";
        }
        fmt::print("  {:<{}}  {:<{}}  {} (one of: {})\n", fmt::format("--{} <{}>", long_name, keys), pad_long, shorts, pad_short, help,
                   keys);
    }

    Enum const &get() const { return bound ? *bound : value; }
    std::string to_string() const {
        for (auto &kv : mapping)
            if ((bound ? *bound : value) == kv.second)
                return kv.first;
        return "";
    }
};

// -------------------------- Alias ---------------------------------------- //

struct Alias : OptionBase {
    OptionBase                *target = nullptr;
    std::optional<std::string> preset_value; // if set, provides value to target

    Alias(StringRef longName, std::initializer_list<char> shorts, OptionBase &tgt, StringRef helpText, OptionCategory &cat,
          std::optional<std::string> value = std::nullopt)
        : OptionBase(longName, shorts, helpText, &cat), target(&tgt), preset_value(std::move(value)) {}

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        seen_cli                          = !from_config;
        seen_config                       = from_config; // forward source
        std::optional<std::string_view> v = preset_value ? std::optional{std::string_view(*preset_value)} : val;
        return target->parse_token(target->long_name, v, error, from_config);
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.pop_back(), shorts.pop_back();
        fmt::print("  {:<{}}  {:<{}}  {} (alias for --{})\n", fmt::format("--{}", long_name), pad_long, shorts, pad_short, help,
                   target ? target->long_name : "?");
    }
};

// -------------------------- Built-in help/version ------------------------ //

struct Builtins {
    OptionCategory cat{"Help"};
    Flag           help{"help", {'h'}, "Show this help message and exit", cat};
    Flag           version{"version", {}, "Show version and exit", cat};
    Builtins() {
        help.value_expected    = ValueExpected::ValueDisallowed;
        version.value_expected = ValueExpected::ValueDisallowed;
    }
};

inline Builtins &builtins() {
    static Builtins B;
    return B;
}

// -------------------------- Config reader -------------------------------- //

// Very small config reader: supports either
//  - key=value lines (ignores blank lines and lines starting with '#')
//  - flat JSON object: { "key": value, ... } where value is string/number/bool
// Returns map of lowercase keys to raw string values.
inline std::map<std::string, std::string, std::less<>> read_config(std::string_view path) {
    std::map<std::string, std::string, std::less<>> kv;
    if (path.empty())
        return kv;
    FILE *f = fopen(std::string(path).c_str(), "rb");
    if (!f)
        return kv;
    std::string buf;
    char        tmp[4096];
    size_t      n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0)
        buf.append(tmp, tmp + n);
    fclose(f);
    auto trim = [](std::string &s) {
        size_t a = 0;
        while (a < s.size() && std::isspace((unsigned char)s[a]))
            ++a;
        size_t b = s.size();
        while (b > a && std::isspace((unsigned char)s[b - 1]))
            --b;
        s = s.substr(a, b - a);
    };
    auto lower = [](std::string s) {
        for (auto &c : s)
            c = (char)std::tolower((unsigned char)c);
        return s;
    };

    auto looks_json = [](std::string const &s) {
        for (char c : s) {
            if (!std::isspace((unsigned char)c))
                return c == '{';
        }
        return false;
    };

    if (!looks_json(buf)) {
        // key=value lines
        size_t start = 0;
        while (start <= buf.size()) {
            size_t      end  = buf.find_first_of("\n", start);
            std::string line = (end == std::string::npos) ? buf.substr(start) : buf.substr(start, end - start);
            start            = (end == std::string::npos) ? buf.size() + 1 : end + 1;
            if (line.empty() || line[0] == '#')
                continue;
            auto eq = line.find('=');
            if (eq == std::string::npos)
                continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            trim(k);
            trim(v);
            kv[lower(k)] = v;
        }
        return kv;
    }

    // Minimal flat JSON parser
    size_t i    = 0;
    auto   s    = buf;
    auto   skip = [&] {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
            ++i;
    };
    i = 0;
    skip();
    if (i >= s.size() || s[i] != '{')
        return kv;
    ++i; // '{'
    while (true) {
        skip();
        if (i < s.size() && s[i] == '}') {
            ++i;
            break;
        }
        if (i >= s.size() || s[i] != '"')
            break;
        ++i;
        std::string key;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                key.push_back(s[i + 1]);
                i += 2;
            } else {
                key.push_back(s[i++]);
            }
        }
        if (i < s.size())
            ++i;
        skip();
        if (i >= s.size() || s[i] != ':')
            break;
        ++i; // ':'
        skip();
        std::string val;
        if (i < s.size() && s[i] == '"') {
            ++i;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    val.push_back(s[i + 1]);
                    i += 2;
                } else {
                    val.push_back(s[i++]);
                }
            }
            if (i < s.size())
                ++i;
        } else if (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' || s[i] == '+')) {
            size_t j = i;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.' || s[j] == 'e' || s[j] == 'E' ||
                                    s[j] == '+' || s[j] == '-'))
                ++j;
            val = s.substr(i, j - i);
            i   = j;
        } else if (s.compare(i, 4, "true") == 0) {
            val = "true";
            i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
            val = "false";
            i += 5;
        } else {
            while (i < s.size() && s[i] != ',' && s[i] != '}')
                ++i;
        }
        std::string lk;
        for (char c : key)
            lk.push_back((char)std::tolower(static_cast<unsigned char>(c)));
        kv[lk] = val;
        skip();
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        skip();
        if (i < s.size() && s[i] == '}') {
            ++i;
            break;
        }
    }
    return kv;
}

// -------------------------- Parsing engine ------------------------------- //

namespace detail {
inline OptionBase *find_long(std::string_view name, Subcommand *current) {
    for (auto *o : Registry::instance().options)
        if (!o->is_positional && o->long_name == name && o->active_for(current))
            return o;
    return nullptr;
}

inline OptionBase *find_short(char c, Subcommand *current) {
    for (auto *o : Registry::instance().options)
        if (!o->is_positional && o->active_for(current))
            for (char s : o->short_names)
                if (s == c)
                    return o;
    return nullptr;
}

inline std::vector<OptionBase *> positional_options(Subcommand *current) {
    std::vector<OptionBase *> v;
    for (auto *o : Registry::instance().options)
        if (o->is_positional && o->active_for(current))
            v.push_back(o);
    return v;
}

} // namespace detail

inline void print_version(std::string_view prog, std::string_view ver) {
    if (!ver.empty())
        fmt::print("{} {}\n", prog, ver);
}

inline void print_help(std::string_view prog, Subcommand *current) {
    auto &R = Registry::instance();

    size_t pad_long = 0, pad_short = 0;
    for (auto *o : R.options)
        if (!o->is_positional && o->visibility == Visibility::Normal && o->active_for(current)) {
            pad_long = std::max(pad_long, std::string("--" + o->long_name).size());
            std::string shorts;
            for (char c : o->short_names)
                shorts += fmt::format("-{}, ", c);
            if (!shorts.empty())
                shorts.pop_back(), shorts.pop_back();
            pad_short = std::max(pad_short, shorts.size());
        }

    if (current && !current->is_root)
        fmt::print("Usage: {} {} [options]", prog, current->name);
    else
        fmt::print("Usage: {} [options]", prog);
    auto pos = detail::positional_options(current);
    for (auto *p : pos)
        fmt::print(" <{}>", p->long_name);
    fmt::print("\n\n");

    if (!current || current->is_root) {
        bool any = false;
        for (auto *sc : R.subcommands)
            if (!sc->is_root) {
                any = true;
                break;
            }
        if (any) {
            fmt::print("Subcommands:\n");
            for (auto *sc : R.subcommands)
                if (!sc->is_root)
                    fmt::print("  {}	{}\n", sc->name, sc->help);
            fmt::print("\n");
        }
    }

    std::map<std::string, std::vector<OptionBase *>> groups;
    for (auto *o : R.options)
        if (!o->is_positional && o->active_for(current))
            groups[o->category ? o->category->name : std::string{}].push_back(o);

    for (auto &[cat, opts] : groups) {
        if (!cat.empty())
            fmt::print("{}:\n", cat);
        for (auto *o : opts)
            o->print_help_line(prog, pad_long + 2, pad_short + 2);
        fmt::print("\n");
    }

    if (!pos.empty()) {
        fmt::print("Positional arguments:\n");
        for (auto *p : pos)
            p->print_help_line(prog, pad_long + 2, 0);
        fmt::print("\n");
    }
}

struct ParseState {
    Subcommand *chosen = &Subcommand::Root();
};

inline ParseResult parse_internal(int argc, char **argv, char const *programName, std::string_view version,
                                  std::map<std::string, std::string, std::less<>> *config) {
    (void)builtins();
    std::string prog = programName ? programName : (argc > 0 && argv && argv[0] ? argv[0] : "program");

    for (auto *o : Registry::instance().options)
        o->finalize_default();

    ParseState st;

    int argi = 1;
    if (argi < argc) {
        std::string first = argv[argi];
        for (auto *sc : Registry::instance().subcommands)
            if (!sc->is_root && sc->name == first) {
                st.chosen = sc;
                ++argi;
                break;
            }
    }

    if (config && !config->empty()) {
        for (auto *o : Registry::instance().options) {
            if (!o->active_for(st.chosen) || o->is_positional)
                continue;
            auto it = config->find(o->long_name);
            if (it == config->end())
                continue;
            std::string                     err;
            std::optional<std::string_view> v;
            if (!it->second.empty())
                v = std::string_view(it->second);
            if (!o->parse_token(o->long_name, v, err, /*from_config=*/true)) {
                fmt::print(stderr, "config error for '{}': {}\n", o->long_name, err);
                return {false, 1};
            }
        }
    }

    auto consume_positional = [&](std::string_view token, std::string &err) -> bool {
        auto                       pos       = detail::positional_options(st.chosen);
        static thread_local size_t pos_index = 0;
        if (pos_index >= pos.size()) {
            err = fmt::format("unexpected positional '{}'", token);
            return false;
        }
        OptionBase *p  = pos[pos_index];
        bool        ok = p->parse_token(p->long_name, token, err);
        if (!ok)
            return false;
        p->seen_cli = true;
        ++p->occurrences;
        if (dynamic_cast<List<std::string> *>(p) == nullptr)
            ++pos_index;
        return true;
    };

    for (int i = argi; i < argc; ++i) {
        std::string_view tok(argv[i]);

        if (tok == "--") {
            while (++i < argc) {
                std::string err;
                if (!consume_positional(argv[i], err)) {
                    fmt::print(stderr, "error: {}\n", err);
                    return {false, 1};
                }
            }
            break;
        }

        if (tok.size() >= 2 && tok[0] == '-' && tok[1] == '-') {
            auto             eq   = tok.find('=');
            std::string_view name = tok.substr(2, eq == std::string_view::npos ? tok.size() - 2 : eq - 2);
            OptionBase      *o    = detail::find_long(name, st.chosen);
            if (!o) {
                fmt::print(stderr, "error: unknown option '--{}'\n", name);
                return {false, 1};
            }
            std::optional<std::string_view> val;
            if (eq != std::string_view::npos)
                val = tok.substr(eq + 1);
            else if (o->value_expected == ValueExpected::ValueRequired) {
                if (i + 1 >= argc) {
                    fmt::print(stderr, "error: option '--{}' requires a value\n", name);
                    return {false, 1};
                }
                val = std::string_view(argv[++i]);
            }
            std::string err;
            if (!o->parse_token(name, val, err)) {
                fmt::print(stderr, "error: {}\n", err);
                return {false, 1};
            }
            if (o->on_seen)
                o->on_seen();
            if (o == &builtins().help) {
                print_help(prog, st.chosen);
                return {false, 0};
            }
            if (o == &builtins().version) {
                print_version(prog, version);
                return {false, 0};
            }
            continue;
        }

        if (tok.size() >= 2 && tok[0] == '-') {
            for (size_t j = 1; j < tok.size(); ++j) {
                char        c = tok[j];
                OptionBase *o = detail::find_short(c, st.chosen);
                if (!o) {
                    fmt::print(stderr, "error: unknown option '-{}'\n", c);
                    return {false, 1};
                }
                std::optional<std::string_view> val;
                bool                            last_in_bundle = (j + 1 == tok.size());
                if (o->value_expected == ValueExpected::ValueRequired) {
                    if (!last_in_bundle) {
                        val = tok.substr(j + 1);
                        j   = tok.size();
                    } else {
                        if (i + 1 >= argc) {
                            fmt::print(stderr, "error: option '-{}' requires a value\n", c);
                            return {false, 1};
                        }
                        val = std::string_view(argv[++i]);
                    }
                }
                std::string err;
                if (!o->parse_token(std::string_view(&c, 1), val, err)) {
                    fmt::print(stderr, "error: {}\n", err);
                    return {false, 1};
                }
                if (o->on_seen)
                    o->on_seen();
                if (o == &builtins().help) {
                    print_help(prog, st.chosen);
                    return {false, 0};
                }
                if (o == &builtins().version) {
                    print_version(prog, version);
                    return {false, 0};
                }
            }
            continue;
        }

        std::string err;
        if (!consume_positional(tok, err)) {
            fmt::print(stderr, "error: {}\n", err);
            return {false, 1};
        }
    }

    for (auto *o : Registry::instance().options)
        if (o->active_for(st.chosen)) {
            if ((o->occurrence == Occurrence::Required || o->occurrence == Occurrence::OneOrMore) && o->occurrences == 0) {
                fmt::print(stderr, "error: missing required option '--{}'\n", o->long_name);
                return {false, 1};
            }
            std::string err;
            if (!o->validate(err)) {
                fmt::print(stderr, "error: {}\n", err);
                return {false, 1};
            }
        }

    return {true, 0};
}

inline ParseResult parse(int argc, char **argv, char const *programName = nullptr, std::string_view version = {}) {
    return parse_internal(argc, argv, programName, version, nullptr);
}

inline ParseResult parse_with_config(int argc, char **argv, char const *programName = nullptr, std::string_view version = {},
                                     std::string_view config_path = {}) {
    auto kv = read_config(config_path);
    return parse_internal(argc, argv, programName, version, &kv);
}

// -------------------------- Convenience builders ------------------------- //

inline Range RangeBetween(long long min_v, long long max_v) {
    return Range{min_v, max_v};
}

} // namespace einsums::cl
