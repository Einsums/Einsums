#include <Einsums/CommandLine/CommandLine.hpp>

namespace einsums::cl {

bool parse_value(std::string_view const &sv, bool &out, std::string const &error_prefix) {
    if (sv == "1" || sv == "true" || sv == "on" || sv == "yes") {
        out = true;
        return true;
    }
    if (sv == "0" || sv == "false" || sv == "off" || sv == "no") {
        out = false;
        return true;
    }
    fmt::print(stderr, "{}invalid boolean '{}', expected true/false", error_prefix, sv);
    return false;
}
bool parse_value(std::string_view const &sv, int &out, std::string const &error_prefix) {
    auto *b = sv.data();
    auto *e = b + sv.size();
    int   v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    fmt::print(stderr, "{}invalid integer '{}'", error_prefix, sv);
    return false;
}
bool parse_value(std::string_view const &sv, long &out, std::string const &error_prefix) {
    auto     *b = sv.data();
    auto     *e = b + sv.size();
    long long v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    fmt::print(stderr, "{}invalid integer '{}'", error_prefix, sv);
    return false;
}
bool parse_value(std::string_view const &sv, long long &out, std::string const &error_prefix) {
    auto     *b = sv.data();
    auto     *e = b + sv.size();
    long long v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    fmt::print(stderr, "{}invalid integer '{}'", error_prefix, sv);
    return false;
}
bool parse_value(std::string_view const &sv, double &out, std::string const &error_prefix) {
    try {
        size_t idx = 0;
        // sv.data: noexcept
        // sv.size: noexcept
        // alloc.allocate: bad_alloc
        std::string tmp(sv);
        double      v = std::stod(tmp, &idx);
        if (idx == tmp.size()) {
            out = v;
            return true;
        }
    } catch (std::bad_alloc &e) {
        std::perror("Could not allocate string for parsing!");
        std::rethrow_exception(std::current_exception());
    } catch (std::invalid_argument &e) {
        fmt::print(stderr, "{}invalid real '{}': could not perform conversion.", error_prefix, sv);
    } catch (std::out_of_range &e) {
        fmt::print(stderr, "{}invalid real '{}': value falls outside of the range of valid values for doubles.", error_prefix, sv);
    } catch (...) {
        fmt::print(stderr, "{}invalid real '{}': unknown problem.", error_prefix, sv);
        std::rethrow_exception(std::current_exception());
    }

    return false;
}

EINSUMS_SINGLETON_IMPL(Builtins)

Builtins::Builtins() {
    help_.set_expected(ValueExpected::ValueDisallowed);
    version_.set_expected(ValueExpected::ValueDisallowed);
}

Builtins::~Builtins() {
    std::fprintf(stderr, "Finalizing Builtins.\n");
}

OptionCategory const &Builtins::get_help_category() const noexcept {
    return cat_;
}
OptionCategory &Builtins::get_help_category() noexcept {
    return cat_;
}

Flag const &Builtins::get_help_flag() const noexcept {
    return help_;
}
Flag &Builtins::get_help_flag() noexcept {
    return help_;
}

Flag const &Builtins::get_version_flag() const noexcept {
    return version_;
}
Flag &Builtins::get_version_flag() noexcept {
    return version_;
}

void print_version(std::string_view prog, std::string_view ver) {
    if (!ver.empty()) {
        fmt::print("{} {}\n", prog, ver);
    }
}

namespace detail {
OptionBase *find_long(std::string_view name) {
    for (auto *o : Registry::get_singleton().get_options()) {
        if (!o->is_positional() && o->get_long_name() == name) {
            return o;
        }
    }
    return nullptr;
}
OptionBase *find_short(char c) {
    for (auto *o : Registry::get_singleton().get_options()) {
        if (!o->is_positional()) {
            for (char s : o->get_short_names()) {
                if (s == c) {
                    return o;
                }
            }
        }
    }
    return nullptr;
}
std::vector<OptionBase *> positional_options() {
    std::vector<OptionBase *> v;
    for (auto *o : Registry::get_singleton().get_options()) {
        if (o->is_positional()) {
            v.push_back(o);
        }
    }
    return v;
}
} // namespace detail

void print_help(std::string_view prog) {
    auto &R = Registry::get_singleton();

    size_t pad_long = 0, pad_short = 0;
    for (auto const *o : R.get_options()) {
        if (!o->is_positional() && o->get_visibility() == Visibility::Normal) {
            pad_long = std::max(pad_long, o->get_long_name().size() + 2 /* +2 for "--" */);
            std::string shorts;
            for (char c : o->get_short_names()) {
                shorts += einsums::format("-{}, ", c);
            }
            if (!shorts.empty()) {
                shorts.erase(shorts.end() - 2, shorts.end());
            }
            pad_short = std::max(pad_short, shorts.size());
        }
    }

    fmt::print("Usage: {} [options]", prog);
    auto pos = detail::positional_options();
    for (auto *p : pos) {
        fmt::print(" <{}>", p->get_long_name());
    }
    fmt::print("\n\n");

    std::map<std::string, std::vector<OptionBase const *>> groups;
    for (auto const *o : R.get_options()) {
        if (!o->is_positional()) {
            groups[o->get_category() ? o->get_category()->get_name() : std::string{}].push_back(o);
        }
    }

    for (auto const &[cat, opts] : groups) {
        if (!cat.empty()) {
            fmt::print("{}:\n", cat);
        }
        for (auto const *o : opts) {
            o->print_help_line(prog, pad_long + 2, pad_short + 2);
        }
        fmt::print("\n");
    }

    if (!pos.empty()) {
        fmt::print("Positional arguments:\n");
        for (auto const *p : pos) {
            p->print_help_line(prog, pad_long + 2, 0);
        }
        fmt::print("\n");
    }
}

} // namespace einsums::cl