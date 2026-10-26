//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Compat/FmtAlternatives.hpp>
#include <Einsums/StringUtil/StringOps.hpp>
#include <Einsums/StringUtil/Trim.hpp>
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

struct StringRef {
  public:
    constexpr StringRef() = default;
    constexpr StringRef(char const *c) : s_(c) {}
    constexpr StringRef(std::string_view v) : s_(v) {}
    constexpr ~StringRef() = default; // This might also need to be constexpr for Windows.
    constexpr operator std::string_view() const noexcept { return s_; }

    [[nodiscard]] constexpr std::string_view const &get_view() const noexcept { return s_; }

    [[nodiscard]] constexpr std::string_view &get_view() noexcept { return s_; }

  private:
    std::string_view s_;
};

struct NumericRange {
  public:
    constexpr NumericRange() noexcept : min_v{(std::numeric_limits<long long>::min)()}, max_v{(std::numeric_limits<long long>::max)()} {}

    constexpr NumericRange(long long min, long long max) noexcept : min_v{min}, max_v{max} {}

    constexpr ~NumericRange() = default;

    [[nodiscard]] constexpr long long min() const noexcept { return min_v; }

    [[nodiscard]] constexpr long long max() const noexcept { return max_v; }

  private:
    long long min_v;
    long long max_v;
};

struct ParseResult {
    bool ok        = true;
    int  exit_code = 0;
};

// Forward decls
struct OptionBase;
struct OptionCategory;
struct ExclusiveCategory;

// -------------------------- Registry ------------------------------------- //

struct EINSUMS_EXPORT Registry final {
    EINSUMS_SINGLETON_DEF(Registry)

  public:
    ~Registry();
  
    void add_option(OptionBase *o);
    void add_category(OptionCategory *c);
    void add_exclusion(ExclusiveCategory *c);

    void clear_for_tests();

    [[nodiscard]] std::list<OptionBase *> const &get_options() const noexcept;
    [[nodiscard]] std::list<OptionBase *>       &get_options() noexcept;

    [[nodiscard]] std::list<OptionCategory *> const &get_categories() const noexcept;
    [[nodiscard]] std::list<OptionCategory *>       &get_categories() noexcept;

    [[nodiscard]] std::list<ExclusiveCategory *> const &get_exclusions() const noexcept;
    [[nodiscard]] std::list<ExclusiveCategory *>       &get_exclusions() noexcept;

  private:
    Registry();
    std::list<OptionBase *>        *options{nullptr};
    std::list<OptionCategory *>    *categories{nullptr};
    std::list<ExclusiveCategory *> *exclusions{nullptr};
};

// -------------------------- Categories ----------------------------------- //

struct EINSUMS_EXPORT OptionCategory {
  public:
    explicit OptionCategory(StringRef const &n);
    
    ~OptionCategory();

    [[nodiscard]] std::string const &get_name() const;
    [[nodiscard]] std::string       &get_name();

  private:
    std::string name;
};

struct ExclusiveCategory;

// -------------------------- Parsing helpers ------------------------------ //

template <typename T>
[[nodiscard]] bool parse_value(std::string_view const &, T &out, std::string const &error_prefix);

// Leave inline because Windows doesn't want us modifying strings.
[[nodiscard]] inline bool parse_value(std::string_view const &sv, std::string &out, [[maybe_unused]] std::string const &error_prefix) {
    out.assign(sv.begin(), sv.end());
    return true;
}
[[nodiscard]] bool EINSUMS_EXPORT parse_value(std::string_view const &sv, bool &out, std::string const &error_prefix);
[[nodiscard]] bool EINSUMS_EXPORT parse_value(std::string_view const &sv, int &out, std::string const &error_prefix);
[[nodiscard]] bool EINSUMS_EXPORT parse_value(std::string_view const &sv, long &out, std::string const &error_prefix);
[[nodiscard]] bool EINSUMS_EXPORT parse_value(std::string_view const &sv, long long &out, std::string const &error_prefix);
[[nodiscard]] bool EINSUMS_EXPORT parse_value(std::string_view const &sv, double &out, std::string const &error_prefix);

// -------------------------- Core types ----------------------------------- //

enum struct Visibility : uint8_t { Normal, Hidden };
enum struct Occurrence : uint8_t { Optional, Required, ZeroOrMore, OneOrMore };
enum struct ValueExpected : uint8_t { ValueDisallowed, ValueOptional, ValueRequired };

struct EINSUMS_EXPORT Positional {};

// Base option (no subcommand affinity)
namespace detail {
struct EINSUMS_EXPORT OptionBaseImpl {
    ~OptionBaseImpl();
    std::string           canonical_name_;
    std::function<void()> on_seen_;
    OptionCategory       *category_    = nullptr;
    ExclusiveCategory    *exclusions_  = nullptr;
    Occurrence            occurrence_  = Occurrence::Optional;
    bool                  seen_cli_    = false;
    bool                  seen_config_ = false;
    int                   occurrences_ = 0;
};
} // namespace detail

struct EINSUMS_EXPORT OptionBase {
  public:
    OptionBase(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, OptionCategory *cat);

    OptionBase(StringRef positional_name, Positional, StringRef helpText);

    OptionBase(StringRef longName, std::initializer_list<char> shorts, StringRef helpText,
               std::shared_ptr<detail::OptionBaseImpl> const &impl);

    virtual ~OptionBase();

    [[nodiscard]] virtual bool parse_token(std::string_view key, std::optional<std::string_view> val, std::string const &error_prefix,
                                           bool from_config = false) = 0;

    virtual void print_help_line(std::string_view prog, size_t pad_long, size_t pad_short) const = 0;

    // Size should include space for the zero terminator.
    [[nodiscard]] virtual bool validate(std::string const &error_prefix) const;

    virtual void finalize_default();

    [[nodiscard]] std::string const &get_long_name() const noexcept;
    [[nodiscard]] std::string       &get_long_name() noexcept;

    [[nodiscard]] std::string const &get_help() const noexcept;
    [[nodiscard]] std::string       &get_help() noexcept;

    [[nodiscard]] std::vector<char> const &get_short_names() const noexcept;
    [[nodiscard]] std::vector<char>       &get_short_names() noexcept;

    [[nodiscard]] OptionCategory const *get_category() const noexcept;
    [[nodiscard]] OptionCategory       *get_category() noexcept;

    [[nodiscard]] ExclusiveCategory const *get_exlusions() const noexcept;
    [[nodiscard]] ExclusiveCategory       *get_exclusions() noexcept;

    [[nodiscard]] Visibility get_visibility() const noexcept;

    [[nodiscard]] Occurrence get_occurrence() const noexcept;

    [[nodiscard]] ValueExpected get_value_expected() const noexcept;

    [[nodiscard]] bool is_positional() const noexcept;

    [[nodiscard]] bool is_from_cli() const noexcept;

    [[nodiscard]] bool is_from_config() const noexcept;

    [[nodiscard]] int num_occurrences() const noexcept;

    void set_long_name(std::string const &name);

    template <typename Iter, typename Traits = std::iterator_traits<std::remove_cvref_t<Iter>>>
        requires(std::is_same_v<typename Traits::value_type, char>)
    void set_short_names(Iter begin, Iter end) {
        short_names_ = std::vector<char>(begin, end);
    }

    template <typename Iter, typename Traits = std::iterator_traits<std::remove_cvref_t<Iter>>>
        requires(std::is_same_v<typename Traits::value_type, char>)
    void add_short_names(Iter begin, Iter end) {
        short_names_.insert(short_names_.end(), begin, end);
    }

    void set_category(OptionCategory *category) noexcept;

    void set_exclusions(ExclusiveCategory *exclusions) noexcept;

    void set_visibility(Visibility vis) noexcept;

    void set_occurrence(Occurrence occurrence) noexcept;

    void set_expected(ValueExpected expected) noexcept;

    void set_positional(bool positional) noexcept;

    void seen_on_cli(int count = 1) noexcept;

    void seen_in_config() noexcept;

    template <class F>
        requires requires(F f) { f(); }
    void set_on_seen(F &&func) {
        pointer_to_impl_->on_seen_ = func;
    }

    std::shared_ptr<detail::OptionBaseImpl> get_impl() noexcept;

    std::shared_ptr<detail::OptionBaseImpl const> get_impl() const noexcept;

  private:
    std::string       long_name_;   // "--long"
    std::vector<char> short_names_; // {'v'}
    std::string       help_;

    std::shared_ptr<detail::OptionBaseImpl> pointer_to_impl_;
    Visibility                              visibility_     = Visibility::Normal;
    ValueExpected                           value_expected_ = ValueExpected::ValueOptional;
    bool                                    is_positional_  = false;
};

struct EINSUMS_EXPORT ExclusiveCategory {
    explicit ExclusiveCategory();
    
    ~ExclusiveCategory();

    [[nodiscard]] bool verify_exclusions() const;

    // Inline for Windows compatibility (can't return anything with heap memory from a dll)
    [[nodiscard]] inline std::list<OptionBase const *> found_options() const {
        std::list<OptionBase const *> out;
        for (auto const *opt : options) {
            if (opt->is_from_cli() || opt->is_from_config() || opt->num_occurrences() > 0) {
                out.push_back(opt);
            }
        }

        return out;
    }

    [[nodiscard]] inline std::list<OptionBase *> found_options() {
        std::list<OptionBase *> out;
        for (auto *opt : options) {
            if (opt->is_from_cli() || opt->is_from_config() || opt->num_occurrences() > 0) {
                out.push_back(opt);
            }
        }

        return out;
    }

    void add_option(OptionBase *option);

  private:
    std::list<OptionBase *> options;
};

// -------------------------- Location & Setter ---------------------------- //

template <typename T>
struct Location {
    T *ptr = nullptr;
    explicit constexpr Location(T &r) noexcept : ptr(&r) {}
};

template <typename T>
using Setter = std::function<void(T const &)>;

// -------------------------- Named-arg tags ------------------------------- //

template <typename T>
struct DefaultTag {
    T v;
};
template <typename T>
struct ImplicitValueTag {
    T v;
};

// template <typename T> DefaultTag(std::decay_t<T>)->DefaultTag<std::decay_t<T>>;
// template <typename T> ImplicitValueTag(std::decay_t<T>)->ImplicitValueTag<std::decay_t<T>>;

template <typename T>
constexpr auto Default(T &&v) -> DefaultTag<std::decay_t<T>> {
    return {std::forward<T>(v)};
}

template <typename T>
constexpr auto ImplicitValue(T &&v) -> ImplicitValueTag<std::decay_t<T>> {
    return {std::forward<T>(v)};
}

struct ValueNameTag {
    std::string name;
};
constexpr ValueNameTag ValueName(std::string const &n) {
    return {n};
}

// -------------------------- Flag ---------------------------------------- //

struct EINSUMS_EXPORT Flag : public OptionBase {
    template <class... Args>
    Flag(StringRef const &longName, std::initializer_list<char> shorts, StringRef const &helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        set_expected(ValueExpected::ValueOptional);
        (apply_arg(std::forward<Args>(args)), ...);
    }

    Flag &on_set(std::function<void(bool const &)> f);

    void finalize_default() override;

    [[nodiscard]] bool parse_token(std::string_view, std::optional<std::string_view> val, std::string const &error_prefix,
                                   bool from_config = false) override;

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override;

    [[nodiscard]] bool get() const noexcept;

    void do_set_on_unseen(bool do_it) noexcept;

    void set_value(bool value) noexcept;

    void set_result(bool value);

  private:
    void apply_arg(OptionCategory &c) noexcept;
    void apply_arg(Visibility v) noexcept;
    void apply_arg(Occurrence o) noexcept;
    void apply_arg(Location<bool> loc) noexcept;
    void apply_arg(std::function<void(bool const &)> f);
    void apply_arg(DefaultTag<bool> d) noexcept;
    void apply_arg(ImplicitValueTag<bool> d) noexcept;
    void apply_arg(ExclusiveCategory &cat);
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Flag");
    }

    bool                             *bound_ = nullptr;
    std::function<void(bool const &)> setter_;
    bool                              value_                 = false;
    bool                              implicit_on_           = true;
    bool                              has_implicit_override_ = false;
    bool                              set_on_unseen_         = true;
};

// Inline for Windows compatibility.
inline std::unique_ptr<ExclusiveCategory> make_yes_no(Flag &yes_flag, Flag &no_flag, bool default_value = false) {
    auto out = std::make_unique<ExclusiveCategory>();

    yes_flag.set_exclusions(out.get());
    no_flag.set_exclusions(out.get());

    out->add_option(&yes_flag);
    out->add_option(&no_flag);

    yes_flag.do_set_on_unseen(false);
    no_flag.do_set_on_unseen(false);

    yes_flag.set_value(true);
    no_flag.set_value(false);

    if (default_value) {
        yes_flag.set_result(true);
    } else {
        no_flag.set_result(true);
    }

    return out;
}

// -------------------------- Opt<T> -------------------------------------- //

template <typename T>
struct Opt : public OptionBase {
  public:
    // With positional default value
    template <class... Args>
    Opt(StringRef const &longName, std::initializer_list<char> shorts, T defaultValue, StringRef const &helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value_(defaultValue) {
        has_default_ = true;
        set_expected(ValueExpected::ValueRequired);
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Without positional default value
    template <class... Args>
    Opt(StringRef const &longName, std::initializer_list<char> shorts, StringRef const &helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        set_expected(ValueExpected::ValueRequired);
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Fluent
    Opt &implicit(T v) {
        implicit_value_ = std::move(v);
        return *this;
    }
    Opt &value_name(StringRef const &n) {
        value_name_ = std::string(n.get_view());
        return *this;
    }
    Opt &on_set(std::function<void(T const &)> f) {
        setter_ = std::move(f);
        return *this;
    }

    void finalize_default() override { set_result(value_); }

    template <typename U = T>
    [[nodiscard]] bool assign_checked(T const &tmp, std::string const &error_prefix, [[maybe_unused]] bool from_config) {
        if constexpr (std::is_arithmetic_v<U>) {
            if (range_.has_value()) {
                long long vll = static_cast<long long>(tmp);
                if (vll < range_->min() || vll > range_->max()) {
                    fmt::print(stderr, "{}value for '--{}' out of range [{}, {}]\n", error_prefix, get_long_name(), range_->min(),
                               range_->max());
                    return false;
                }
            }
        }
        value_ = tmp;
        set_result(value_);
        return true;
    }

    [[nodiscard]] bool parse_token(std::string_view, std::optional<std::string_view> val, std::string const &error_prefix,
                                   bool from_config = false) override {
        if (!val.has_value()) {
            if (get_value_expected() == ValueExpected::ValueRequired) {
                if (implicit_value_.has_value()) {
                    if (!assign_checked(*implicit_value_, error_prefix, from_config)) {
                        return false;
                    }
                } else {
                    fmt::print(stderr, "{}option '--{}' requires a value", error_prefix, get_long_name());
                    return false;
                }
            }
            if (from_config) {
                seen_in_config();
                return true;
            }
            seen_on_cli();
            return true;
        }
        T tmp{};
        if (!parse_value(*val, tmp, error_prefix)) {
            return false;
        }
        if (!assign_checked(tmp, error_prefix, from_config)) {
            return false;
        }
        if (from_config) {
            seen_in_config();
            return true;
        }
        seen_on_cli();
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (get_visibility() == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : get_short_names()) {
            shorts += einsums::format("-{}, ", c);
        }
        if (!shorts.empty()) {
            shorts.erase(shorts.end() - 2, shorts.end());
        }
        std::string def;
        if (has_default_ && !bound_) {
            def = einsums::format(" (default: {})", value_);
        }
        fmt::print("  {:<{}}  {:<{}}  {}{}\n", einsums::format("--{} <{}>", get_long_name(), value_name_), pad_long, shorts, pad_short,
                   get_help(), def);
    }

    [[nodiscard]] T const &get() const noexcept { return bound_ ? *bound_ : value_; }

    void set_result(T const &value) {
        if (bound_) {
            *bound_ = value;
        }

        if (setter_) {
            setter_(value);
        }
    }

  private:
    void apply_arg(OptionCategory &c) noexcept { set_category(&c); }
    void apply_arg(Visibility v) noexcept { set_visibility(v); }
    void apply_arg(Occurrence o) noexcept { set_occurrence(o); }
    void apply_arg(ValueExpected ve) noexcept { set_expected(ve); }
    void apply_arg(NumericRange r) noexcept { range_ = r; }
    void apply_arg(Location<T> loc) noexcept { bound_ = loc.ptr; }
    void apply_arg(std::function<void(T const &)> f) { setter_ = std::move(f); }
    void apply_arg(ValueNameTag t) { value_name_ = std::move(t.name); }
    template <class U>
        requires(std::is_same_v<std::decay_t<U>, T>)
    void apply_arg(DefaultTag<U> d) noexcept {
        value_       = d.v;
        has_default_ = true;
    }
    template <class U>
        requires(std::is_same_v<std::decay_t<U>, T>)
    void apply_arg(ImplicitValueTag<U> d) noexcept {
        implicit_value_ = d.v;
    }
    void apply_arg(ExclusiveCategory &cat) {
        cat.add_option(this);
        set_exclusions(&cat);
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Opt<T> constructor. Check your argument types.");
    }

    T                              value_{};
    T                             *bound_       = nullptr;
    bool                           has_default_ = false;
    std::optional<NumericRange>    range_;
    std::optional<T>               implicit_value_;
    std::string                    value_name_ = "value";
    std::function<void(T const &)> setter_;
};

// -------------------------- List<T> ------------------------------------- //

template <typename T>
struct List : public OptionBase {
  public:
    // Named list: --include a --include b  OR  --include=a,b
    template <class... Args>
    List(StringRef const &longName, std::initializer_list<char> shorts, StringRef const &helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        set_expected(ValueExpected::ValueRequired);
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Positional list (captures remaining tokens)
    List(StringRef const &positional_name, Positional, StringRef const &helpText) : OptionBase(positional_name, Positional{}, helpText) {
        set_positional(true);
        set_expected(ValueExpected::ValueRequired);
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string const &error_prefix,
                     bool from_config = false) override {
        if (!val.has_value()) {
            fmt::print("{}option '--{}' requires a value", error_prefix, get_long_name());
            return false;
        }
        std::string_view s     = *val;
        size_t           start = 0;
        while (start <= s.size()) {
            size_t           comma = s.find(',', start);
            std::string_view item  = (comma == std::string_view::npos) ? s.substr(start) : s.substr(start, comma - start);
            if (!item.empty()) {
                T tmp{};
                if (!parse_value(item, tmp, error_prefix))
                    return false;
                vals_.push_back(std::move(tmp));
            }
            if (comma == std::string_view::npos)
                break;
            start = comma + 1;
        }
        if (from_config) {
            seen_in_config();
            return true;
        }
        seen_on_cli();
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (get_visibility() == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : get_short_names())
            shorts += einsums::format("-{}, ", c);
        if (!shorts.empty())
            shorts.erase(shorts.end() - 2, shorts.end());
        fmt::print("  {:<{}}  {:<{}}  {}\n", einsums::format("--{} <v1,v2,...>", get_long_name()), pad_long, shorts, pad_short, get_help());
    }

    std::vector<T> const &values() const { return vals_; }

  private:
    void apply_arg(OptionCategory &c) noexcept { set_category(&c); }
    void apply_arg(Visibility v) noexcept { set_visibility(v); }
    void apply_arg(Occurrence o) noexcept { set_occurrence(o); }
    void apply_arg(ExclusiveCategory &cat) {
        cat.add_option(this);
        set_exclusions(&cat);
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to List");
    }
    std::vector<T> vals_;
};

// -------------------------- OptEnum ------------------------------------- //

template <typename Enum>
struct OptEnum : OptionBase {
  public:
    template <class... Args>
    OptEnum(StringRef const &longName, std::initializer_list<char> shorts, Enum defaultValue,
            std::initializer_list<std::pair<std::string, Enum>> map, StringRef const &helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value_(defaultValue), has_default_(true), mapping_(map) {
        set_expected(ValueExpected::ValueRequired);
        (apply_arg(std::forward<Args>(args)), ...);
    }

    template <class... Args>
    OptEnum(StringRef const &longName, std::initializer_list<char> shorts, std::initializer_list<std::pair<std::string, Enum>> map,
            StringRef const &helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), mapping_(map) {
        set_expected(ValueExpected::ValueRequired);
        (apply_arg(std::forward<Args>(args)), ...);
    }
    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string const &error_prefix,
                     bool from_config = false) override {
        if (!val.has_value()) {
            fmt::print(stderr, "{}option '--{}' requires a value", error_prefix, get_long_name());
            return false;
        }
        auto it = mapping_.find(std::string(*val));
        if (it == mapping_.end()) {
            std::string keys;
            size_t      i = 0;
            for (auto const &kv : mapping_) {
                keys += kv.first;
                if (++i < mapping_.size())
                    keys += ", ";
            }
            fmt::print(stderr, "{}invalid value '{}' for '--{}' (choices: {})", error_prefix, *val, get_long_name(), keys);
            return false;
        }
        Enum newv = it->second;
        set_result(newv);
        if (from_config) {
            seen_in_config();
            return true;
        }
        seen_on_cli();
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (get_visibility() == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : get_short_names())
            shorts += einsums::format("-{}, ", c);
        if (!shorts.empty())
            shorts.erase(shorts.end() - 2, shorts.end());
        std::string keys;
        size_t      i = 0;
        for (auto const &kv : mapping_) {
            keys += kv.first;
            if (++i < mapping_.size())
                keys += "|";
        }
        fmt::print("  {:<{}}  {:<{}}  {} (one of: {})\n", einsums::format("--{} <{}>", get_long_name(), keys), pad_long, shorts, pad_short,
                   get_help(), keys);
    }

    Enum const &get() const noexcept { return bound_ ? *bound_ : value_; }
    std::string to_string() const {
        for (auto const &kv : mapping_)
            if (get() == kv.second)
                return kv.first;
        return {};
    }

    void set_result(Enum value) {
        if (bound_) {
            *bound_ = value;
        } else {
            value_ = value;
        }

        if (setter_) {
            setter_(value);
        }
    }

  private:
    void apply_arg(OptionCategory &c) noexcept { set_category(&c); }
    void apply_arg(Visibility v) noexcept { set_visibility(v); }
    void apply_arg(Occurrence o) noexcept { set_occurrence(o); }
    void apply_arg(Location<Enum> loc) noexcept { bound_ = loc.ptr; }
    void apply_arg(std::function<void(Enum const &)> f) { setter_ = std::move(f); }
    void apply_arg(ExclusiveCategory &cat) {
        cat.add_option(this);
        set_exclusions(&cat);
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to OptEnum");
    }
    Enum                                     value_{};
    Enum                                    *bound_       = nullptr;
    bool                                     has_default_ = false;
    std::map<std::string, Enum, std::less<>> mapping_;
    std::function<void(Enum const &, bool)>  setter_;
};

// -------------------------- Alias --------------------------------------- //

struct EINSUMS_EXPORT Alias : public OptionBase {
  public:
    template <class... Args>
    Alias(StringRef const &longName, std::initializer_list<char> shorts, OptionBase &tgt, StringRef const &helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, tgt.get_impl()), target_{&tgt} {
        (apply_arg(std::forward<Args>(args)), ...);
    }

    [[nodiscard]] bool parse_token(std::string_view, std::optional<std::string_view> val, std::string const &error,
                                   bool from_config = false) override;

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override;

  private:
    void apply_arg(Visibility v) noexcept;
    void apply_arg(ValueExpected ve) noexcept;
    void apply_arg(std::string const &v);
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Alias");
    }

    OptionBase *target_;

    std::optional<std::string> preset_value_;
};

// -------------------------- Built-ins ----------------------------------- //

struct EINSUMS_EXPORT Builtins final {
    EINSUMS_SINGLETON_DEF(Builtins)
  public:
    ~Builtins();
    [[nodiscard]] OptionCategory const &get_help_category() const noexcept;
    [[nodiscard]] OptionCategory       &get_help_category() noexcept;

    [[nodiscard]] Flag const &get_help_flag() const noexcept;
    [[nodiscard]] Flag       &get_help_flag() noexcept;

    [[nodiscard]] Flag const &get_version_flag() const noexcept;
    [[nodiscard]] Flag       &get_version_flag() noexcept;

  private:
    Builtins();
    OptionCategory cat_{"Help"};
    Flag           help_{"help", {'h'}, "Show this help message and exit", cat_};
    Flag           version_{"version", {}, "Show version and exit", cat_};
};

std::map<std::string, std::string, std::less<>> EINSUMS_EXPORT read_config(std::string_view const &path);

// -------------------------- Help / Version ------------------------------- //

void EINSUMS_EXPORT print_version(std::string_view prog, std::string_view ver);

namespace detail {
EINSUMS_EXPORT OptionBase *find_long(std::string_view name);
EINSUMS_EXPORT OptionBase *find_short(char c);
EINSUMS_EXPORT std::vector<OptionBase *> positional_options();
} // namespace detail

void EINSUMS_EXPORT print_help(std::string_view prog);

// -------------------------- Parser (no subcommands) ---------------------- //

ParseResult EINSUMS_EXPORT parse_internal(std::vector<std::string> const &args, char const *programName, std::string_view version,
                                          std::map<std::string, std::string, std::less<>> const *config,
                                          std::vector<std::string>                              *unknown_args = nullptr);

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
