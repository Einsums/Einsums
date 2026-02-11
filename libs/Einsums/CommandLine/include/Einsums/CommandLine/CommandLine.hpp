//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums::cl {

// -------------------------- Small utilities ------------------------------- //


// Forward decls
struct OptionBase;
struct OptionCategory;
struct ExclusiveCategory;

// -------------------------- Registry ------------------------------------- //

struct Registry final {
    EINSUMS_MAKE_SINGLETON(Registry)
public:
    void add_option(OptionBase *option);
    void add_category(OptionCategory *category);
    void add_exclusion(ExclusiveCategory *category);

    std::list<OptionBase *> const &get_options() const;
    std::list<OptionCategory *> const &get_categories() const;
    std::list<ExclusiveCategory *> const &get_exclusions() const;

    void clear_for_tests();

private:
    std::list<OptionBase *>        options_;
    std::list<OptionCategory *>    categories_;
    std::list<ExclusiveCategory *> exclusions_;

    Registry() = default;
};

// -------------------------- Categories ----------------------------------- //

struct OptionCategory {
public:
    explicit OptionCategory(std::string_view const &name);
    explicit OptionCategory(std::string const &name);
    explicit OptionCategory(char const *name);

    std::string const &get_name() const;

private:
    std::string name;
};

struct ExclusiveCategory;

// -------------------------- Parsing helpers ------------------------------ //

template <typename T>
bool parse_value(std::string_view, T &out, std::string &err);

EINSUMS_EXPORT bool parse_value(std::string_view sv, std::string &out, std::string &);

EINSUMS_EXPORT bool parse_value(std::string_view sv, bool &out, std::string &err);

EINSUMS_EXPORT bool parse_value(std::string_view sv, int &out, std::string &err);

EINSUMS_EXPORT bool parse_value(std::string_view sv, long &out, std::string &err);

EINSUMS_EXPORT bool parse_value(std::string_view sv, long long &out, std::string &err);

EINSUMS_EXPORT bool parse_value(std::string_view sv, double &out, std::string &err);

// -------------------------- Core types ----------------------------------- //

enum struct Visibility : uint8_t { Normal, Hidden };
enum struct Occurrence : uint8_t { Optional, Required, ZeroOrMore, OneOrMore };
enum struct ValueExpected : uint8_t { ValueDisallowed, ValueOptional, ValueRequired };

struct Positional {};

// Base option (no subcommand affinity)
struct OptionBase {
public:
    OptionBase(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, OptionCategory *cat);

    OptionBase(std::string_view positional_name, Positional, std::string_view helpText);

    virtual ~OptionBase() = default;

    virtual bool parse_token(std::string_view const &key, std::optional<std::string_view> val, std::string &error, bool from_config = false) = 0;

    virtual void print_help_line(std::string_view prog, size_t pad_long, size_t pad_short) const = 0;

    virtual bool validate(std::string &error) const;

    virtual void finalize_default();

    std::string const &long_name() const;

    std::vector<char> const &short_name() const;

    std::string const &help() const;

    OptionCategory const *category() const;

    ExclusiveCategory const *exclusions() const;

    Visibility visibility() const;

    Occurrence occurrence() const;

    ValueExpected value_expected() const;

    bool is_positional() const;

    bool seen_cli() const;

    bool seen_config() const;

    int occurences() const;

    void on_seen() const;



protected:
    std::string        long_name_;   // "--long"
    std::vector<char>  short_names_; // {'v'}
    std::string        help_;
    OptionCategory    *category_       = nullptr;
    ExclusiveCategory *exclusions_     = nullptr;
    Visibility         visibility_     = Visibility::Normal;
    Occurrence         occurrence_     = Occurrence::Optional;
    ValueExpected      value_expected_ = ValueExpected::ValueOptional;
    bool               is_positional_  = false;
    bool               seen_cli_       = false;
    bool               seen_config_    = false;
    int                occurrences_    = 0;

    std::function<void()> on_seen_;
};

struct ExclusiveCategory {
public:
    explicit ExclusiveCategory();

    bool verify_exclusions() const;

    std::list<OptionBase *> found_options();

    std::list<OptionBase *> const &options() const;

    void add_option(OptionBase *);

private:
    std::list<OptionBase *> options_;
};

// -------------------------- Location & Setter ---------------------------- //

template <typename T>
struct Location {
    T *ptr = nullptr;
    constexpr explicit Location(T &r) : ptr(&r) {}
};

template <typename T>
struct Setter {
    std::function<void(T const &)> fn;
    constexpr Setter() = default;
    template <typename F>
    constexpr explicit Setter(F &&f) : fn(std::forward<F>(f)) {}
};

// -------------------------- Named-arg tags ------------------------------- //

template <typename T>
struct DefaultTag {
    T v;

    constexpr explicit DefaultTag(T &&v) : v{std::forward<T>(v)} {}
};
template <typename T>
struct ImplicitValueTag {
    T v;

    constexpr explicit ImplicitValueTag(T &&v) : v{std::forward<T>(v)} {}
};

struct ValueNameTag {
    std::string name;

    ValueNameTag(std::string n) : name{std::move(n)} {}
};

// -------------------------- Flag ---------------------------------------- //

struct Flag : OptionBase {
public:
    template <class... Args>
    Flag(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueOptional;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    Flag &on_set(std::function<void(bool)> f);

    void finalize_default() override;

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override;

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override;

    bool get() const;

  private:
    void apply_arg(OptionCategory &c);
    void apply_arg(Visibility v);
    void apply_arg(Occurrence o);
    void apply_arg(Location<bool> loc);
    void apply_arg(std::function<void(bool const &)> f);
    void apply_arg(Setter<bool> s);
    void apply_arg(DefaultTag<bool> d);
    void apply_arg(ImplicitValueTag<bool> d);
    void apply_arg(ExclusiveCategory &cat);
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Flag");
    }

    bool                              value = false;
    bool                             *bound = nullptr;
    std::function<void(bool)> setter;
    bool                              implicit_on           = true;
    bool                              has_implicit_override = false;
    bool                              set_on_unseen         = true;
};

EINSUMS_EXPORT std::shared_ptr<ExclusiveCategory> make_yes_no(Flag &yes_flag, Flag &no_flag, bool default_value = false);

// -------------------------- Opt<T> -------------------------------------- //

template <typename T>
struct Opt : OptionBase {
public:
    // With positional default value
    template <class... Args>
    Opt(std::string_view longName, std::initializer_list<char> shorts, T defaultValue, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value(defaultValue) {
        has_default    = true;
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Without positional default value
    template <class... Args>
    Opt(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Fluent
    Opt &Implicit(T v) {
        implicit_value = std::move(v);
        return *this;
    }
    Opt &ValueName(std::string_view n) {
        value_name = std::string(n);
        return *this;
    }
    Opt &OnSet(std::function<void(T const &)> f) {
        setter = std::move(f);
        return *this;
    }

    void finalize_default() override {
        if (bound)
            *bound = value;
        if (setter)
            setter(value);
    }

    template <typename U = T>
    bool assign_checked(T const &tmp, std::string &error, bool from_config) {
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
        if (setter)
            setter(value);
        return true;
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        if (!val.has_value()) {
            if (value_expected == ValueExpected::ValueRequired) {
                if (implicit_value.has_value()) {
                    if (!assign_checked(*implicit_value, error, from_config))
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
        if (!assign_checked(tmp, error, from_config))
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
            shorts.erase(shorts.end() - 2, shorts.end());
        std::string def;
        if (has_default && !bound)
            def = fmt::format(" (default: {})", value);
        fmt::print("  {:<{}}  {:<{}}  {}{}\n", fmt::format("--{} <{}>", long_name, value_name), pad_long, shorts, pad_short, help, def);
    }

    T const &get() const { return bound ? *bound : value; }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(ValueExpected ve) { value_expected = ve; }
    void apply_arg(Range r) { range = r; }
    void apply_arg(Location<T> loc) { bound = loc.ptr; }
    void apply_arg(std::function<void(T const &)> f) { setter = std::move(f); }
    void apply_arg(Setter<T> s) { setter = s.fn; }
    void apply_arg(ValueNameTag t) { value_name = std::move(t.name); }
    template <class U>
    void apply_arg(DefaultTag<U> d) {
        static_assert(std::is_same_v<std::decay_t<U>, T>, "Default(value) type must match Opt<T>");
        value       = d.v;
        has_default = true;
    }
    template <class U>
    void apply_arg(ImplicitValueTag<U> d) {
        static_assert(std::is_same_v<std::decay_t<U>, T>, "ImplicitValue(value) type must match Opt<T>");
        implicit_value = d.v;
    }
    void apply_arg(ExclusiveCategory &cat) {
        cat.options.push_back(this);
        exclusions = &cat;
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Opt<T> constructor");
    }
    T                              value{};
    T                             *bound       = nullptr;
    bool                           has_default = false;
    std::optional<Range>           range;
    std::optional<T>               implicit_value;
    std::string                    value_name = "value";
    std::function<void(T const &)> setter;
};

// -------------------------- List<T> ------------------------------------- //

template <typename T>
struct List : OptionBase {
    // Named list: --include a --include b  OR  --include=a,b
    template <class... Args>
    List(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Positional list (captures remaining tokens)
    List(std::string_view positional_name, Positional, std::string_view helpText) : OptionBase(positional_name, Positional{}, helpText) {
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
            shorts.erase(shorts.end() - 2, shorts.end());
        fmt::print("  {:<{}}  {:<{}}  {}\n", fmt::format("--{} <v1,v2,...>", long_name), pad_long, shorts, pad_short, help);
    }

    std::vector<T> const &values() const { return vals; }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(ExclusiveCategory &cat) {
        cat.options.push_back(this);
        exclusions = &cat;
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to List");
    }

    std::vector<T> vals;
};

// -------------------------- OptEnum ------------------------------------- //

template <typename Enum>
struct OptEnum : OptionBase {
    template <class... Args>
    OptEnum(std::string_view longName, std::initializer_list<char> shorts, Enum defaultValue,
            std::initializer_list<std::pair<std::string, Enum>> map, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value(defaultValue), has_default(true), mapping(map) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    template <class... Args>
    OptEnum(std::string_view longName, std::initializer_list<char> shorts, std::initializer_list<std::pair<std::string, Enum>> map,
            std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), mapping(map) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
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
        Enum newv = it->second;
        if (bound)
            *bound = newv;
        else
            value = newv;
        if (setter)
            setter(bound ? *bound : value, from_config);
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
            shorts.erase(shorts.end() - 2, shorts.end());
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
        return {};
    }
private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(Location<Enum> loc) { bound = loc.ptr; }
    void apply_arg(std::function<void(Enum const &, bool)> f) { setter = std::move(f); }
    void apply_arg(Setter<Enum> s) { setter = s.fn; }
    void apply_arg(ExclusiveCategory &cat) {
        cat.options.push_back(this);
        exclusions = &cat;
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to OptEnum");
    }

    Enum                                     value{};
    Enum                                    *bound       = nullptr;
    bool                                     has_default = false;
    std::map<std::string, Enum, std::less<>> mapping;
    std::function<void(Enum const &, bool)>  setter;
};

// -------------------------- Alias --------------------------------------- //

struct Alias : OptionBase {
public:
    template <class... Args>
    Alias(std::string_view longName, std::initializer_list<char> shorts, OptionBase &tgt, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), target(&tgt) {
        (apply_arg(std::forward<Args>(args)), ...);
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override;

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override;

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(std::string v) { preset_value = std::move(v); }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Alias");
    }

    OptionBase                *target = nullptr;
    std::optional<std::string> preset_value;
};

// -------------------------- Built-ins ----------------------------------- //

struct Builtins {
    EINSUMS_SINGLETON_DEF(Builtins);
public:
    OptionCategory &category();
    OptionCategory const &category() const;

    Flag &help();
    Flag const &help() const;

    Flag &version();
    Flag const &version() const;

private:
    OptionCategory cat_{"Help"};
    Flag           help_{"help", {'h'}, "Show this help message and exit", cat};
    Flag           version_{"version", {}, "Show version and exit", cat};
    Builtins();
};


// -------------------------- Config reader -------------------------------- //

EINSUMS_EXPORT std::map<std::string, std::string, std::less<>> read_config(std::string_view path);

// -------------------------- Help / Version ------------------------------- //

EINSUMS_EXPORT void print_version(std::string_view prog, std::string_view ver);

namespace detail {
EINSUMS_EXPORT OptionBase *find_long(std::string_view name);
EINSUMS_EXPORT OptionBase *find_short(char c);
EINSUMS_EXPORT std::vector<OptionBase *> positional_options();
} // namespace detail

EINSUMS_EXPORT void print_help(std::string_view prog);

// -------------------------- Parser (no subcommands) ---------------------- //

EINSUMS_EXPORT ParseResult parse_internal(std::vector<std::string> const &args, char const *programName, std::string_view version,
                                  std::map<std::string, std::string, std::less<>> *config,
                                  std::vector<std::string>                        *unknown_args = nullptr);

/**
 * Parses command-line arguments storing their presence into previously registered Opt/Flag/OpenEnum option.
 *
 * @param args command-line arguments converted to a std::vector<std::string>
 * @param programName the program name to display in help printing
 * @param version the program version to display in version printing
 * @param unknown_args arguments not understood by our parser are placed here
 * @return if ParseResult.ok is true then parsing completed successfully
 */
inline ParseResult parse(std::vector<std::string> const &args, char const *programName = nullptr, std::string_view version = {},
                         std::vector<std::string> *unknown_args = nullptr) {
    return parse_internal(args, programName, version, nullptr, unknown_args);
}

/**
 * Parses command-line arguments storing their presence into previously registered Opt/Flag/OpenEnum option.
 *
 * @param args command-line arguments converted to a std::vector<std::string>
 * @param programName the program name to display in help printing
 * @param version the program version to display in version printing
 * @param config_path key=value or simple json config file that you want to be read in before command line processing
 * @param unknown_args arguments not understood by our parser are placed here
 * @return if ParseResult.ok is true then parsing completed successfully
 */
inline ParseResult parse_with_config(std::vector<std::string> const &args, char const *programName = nullptr, std::string_view version = {},
                                     std::string_view config_path = {}, std::vector<std::string> *unknown_args = nullptr) {
    auto kv = read_config(config_path);
    return parse_internal(args, programName, version, &kv, unknown_args);
}

} // namespace einsums::cl
