//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>

namespace einsums::cl {

OptionBase::OptionBase(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, OptionCategory *cat)
        : long_name(longName), short_names(shorts), help(helpText), category(cat) {
    Registry::instance().add_option(this);
}

OptionBase::OptionBase(std::string_view positional_name, Positional, std::string_view helpText)
: long_name(positional_name), help(helpText), is_positional(true) {
    Registry::instance().add_option(this);
}

bool OptionBase::validate(std::string &error) const {
    return true;
}

void OptionBase::finalize_default() {}

std::string const &OptionBase::long_name() const {
    return long_name_;
}

std::vector<char> const &OptionBase::short_name() const {
    return short_name_;
}

std::string const &OptionBase::help() const {
    return help_;
}

OptionCategory const *OptionBase::category() const {
    return category_;
}

ExclusiveCategory const *OptionBase::exclusions() const {
    return exclusions_;
}

Visibility OptionBase::visibility() const {
    return visibility_;
}

Occurrence OptionBase::occurrence() const {
    return occurrence_;
}

ValueExpected OptionBase::value_expected() const {
    return value_expected_;
}

bool OptionBase::is_positional() const {
    return is_positional_;
}

bool OptionBase::seen_cli() const {
    return seen_cli_;
}

bool OptionBase::seen_config() const {
    return seen_config_;
}

int OptionBase::occurrences() const {
    return occurrences_;
}

void OptionBase::on_seen() const {
    on_seen_();
}

Flag &Flag::on_set(std::function<void(bool)> f) {
    setter = std::move(f);
    return *this;
}

void Flag::finalize_default() {
    if (set_on_unseen || occurrences > 0 || seen_cli || seen_config) {
        if (bound) {
            *bound = value;
        }
        if (setter) {
            setter(value);
        }
    }
}

bool Flag::parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config) {
    bool tmp;
    if (!val.has_value()) {
        tmp = has_implicit_override ? implicit_on : true; // presence => true
    } else if (!parse_value(*val, tmp, error)) {
        return false;
    }
    value = tmp;
    if (bound)
        *bound = value;
    if (setter)
        setter(value);
    if (from_config) {
        seen_config = true;
        return true;
    }
    seen_cli = true;
    ++occurrences;
    return true;
}

void Flag::print_help_line(std::string_view, size_t pad_long, size_t pad_short) const {
    if (visibility == Visibility::Hidden)
        return;
    std::string shorts;
    for (char c : short_names)
        shorts += fmt::format("-{}, ", c);
    if (!shorts.empty())
        shorts.erase(shorts.end() - 2, shorts.end());
    fmt::print("  {:<{}}  {:<{}}  {}\n", fmt::format("--{}", long_name), pad_long, shorts, pad_short, help);
}

bool Flag::get() const { return bound ? *bound : value; }

void Flag::apply_arg(OptionCategory &c) { category = &c; }
void Flag::apply_arg(Visibility v) { visibility = v; }
void Flag::apply_arg(Occurrence o) { occurrence = o; }
void Flag::apply_arg(Location<bool> loc) { bound = loc.ptr; }
void Flag::apply_arg(std::function<void(bool const &)> f) { setter = std::move(f); }
void Flag::apply_arg(Setter<bool> s) { setter = s.fn; }
void Flag::apply_arg(DefaultTag<bool> d) {
    value = d.v;
    if (bound)
        *bound = value;
}
void Flag::apply_arg(ImplicitValueTag<bool> d) {
    implicit_on           = d.v;
    has_implicit_override = true;
}
void Flag::apply_arg(ExclusiveCategory &cat) {
    cat.options.push_back(this);
    exclusions = &cat;
}

bool Alias::parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config) {
    seen_cli                          = !from_config;
    seen_config                       = from_config;
    std::optional<std::string_view> v = preset_value ? std::optional{std::string_view(*preset_value)} : val;
    return target->parse_token(target->long_name, v, error, from_config);
}

void Alias::print_help_line(std::string_view, size_t pad_long, size_t pad_short) const {
    if (visibility == Visibility::Hidden)
        return;
    std::string shorts;
    for (char c : short_names)
        shorts += fmt::format("-{}, ", c);
    if (!shorts.empty())
        shorts.erase(shorts.end() - 2, shorts.end());
    fmt::print("  {:<{}}  {:<{}}  {} (alias for --{})\n", fmt::format("--{}", long_name), pad_long, shorts, pad_short, help,
            target ? target->long_name : "?");
}

void Alias::apply_arg(OptionCategory &c) { category = &c; }
void Alias::apply_arg(Visibility v) { visibility = v; }
void Alias::apply_arg(Occurrence o) { occurrence = o; }
void Alias::apply_arg(std::string v) { preset_value = std::move(v); }

EINSUMS_SINGLETON_IMPL(Builtins)

OptionCategory &Builtins::category() {
    return cat_;
}
OptionCategory const &Builtins::category() const {
    return cat_;
}

Flag &Builtins::help() {
    return help_;
}
Flag const &Builtins::help() const {
    return help_;
}

Flag &Builtins::version() {
    return version_;
}
Flag const &Builtins::version() const {
    return version_;
}

Builtins::Builtins() {
    help.value_expected    = ValueExpected::ValueDisallowed;
    version.value_expected = ValueExpected::ValueDisallowed;
}

namespace detail {

OptionBase *find_long(std::string_view name) {
    for (auto *o : Registry::instance().options)
        if (!o->is_positional && o->long_name == name)
            return o;
    return nullptr;
}

OptionBase *find_short(char c) {
    for (auto *o : Registry::instance().options)
        if (!o->is_positional)
            for (char s : o->short_names)
                if (s == c)
                    return o;
    return nullptr;
}

std::vector<OptionBase *> positional_options() {
    std::vector<OptionBase *> v;
    for (auto *o : Registry::instance().options)
        if (o->is_positional)
            v.push_back(o);
    return v;
}

void print_help(std::string_view prog) {
    auto &R = Registry::instance();

    size_t pad_long = 0, pad_short = 0;
    for (auto *o : R.options)
        if (!o->is_positional && o->visibility == Visibility::Normal) {
            pad_long = std::max(pad_long, std::string("--" + o->long_name).size());
            std::string shorts;
            for (char c : o->short_names)
                shorts += fmt::format("-{}, ", c);
            if (!shorts.empty())
                shorts.erase(shorts.end() - 2, shorts.end());
            pad_short = std::max(pad_short, shorts.size());
        }

    fmt::print("Usage: {} [options]", prog);
    auto pos = detail::positional_options();
    for (auto *p : pos)
        fmt::print(" <{}>", p->long_name);
    fmt::print("\n\n");

    std::map<std::string, std::vector<OptionBase *>> groups;
    for (auto *o : R.options)
        if (!o->is_positional)
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

ParseResult parse_internal(std::vector<std::string> const &args, char const *programName, std::string_view version,
                                  std::map<std::string, std::string, std::less<>> *config,
                                  std::vector<std::string>                        *unknown_args) {
    Builtins                 _;
    GlobalConfigMapLockScope __;
    std::string              prog = programName ? programName : (!args.empty() ? args[0] : "Einsums");

    for (auto *o : Registry::instance().options) {
        o->finalize_default();
    }

    // Apply config first (defaults < config < CLI)
    if (config && !config->empty()) {
        for (auto *o : Registry::instance().options) {
            if (o->is_positional)
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
                return CONFIG_ERROR;
            }
        }
    }

    auto looks_like_option_token = [](std::string_view sv) -> bool {
        if (sv.size() >= 1 && sv[0] == '-') {
            // Treat numeric-looking tokens like "-5" or "-3.14" as values, not options
            if (sv.size() >= 2 && std::isdigit(static_cast<unsigned char>(sv[1])))
                return false;
            return true;
        }
        return false;
    };

    size_t pos_index = 0;

    auto consume_positional = [&](std::string_view token, std::string &err) -> bool {
        auto pos = detail::positional_options();
        if (pos_index >= pos.size()) {
            // No positional to consume -> treat as unknown (per your policy)
            if (unknown_args)
                unknown_args->push_back(std::string(token));
            err.clear();
            return true;
        }

        OptionBase *p  = pos[pos_index];
        bool        ok = p->parse_token(p->long_name, token, err);
        if (!ok)
            return false;

        p->seen_cli = true;
        ++p->occurrences;

        // Stay on the same positional if it's a List<std::string>
        // so it can keep capturing subsequent tokens.
        if (dynamic_cast<List<std::string> *>(p) == nullptr) {
            ++pos_index;
        }

        return true;
    };

    // Parse CLI
    for (size_t i = 1; i < args.size(); ++i) {
        std::string_view tok(args[i]);

        // Everything after "--" -> unknown_args
        if (tok == "--") {
            while (++i < args.size()) {
                if (unknown_args)
                    unknown_args->push_back(args[i]);
            }
            break;
        }

        // Long options: --name or --name=value
        if (tok.size() >= 2 && tok[0] == '-' && tok[1] == '-') {
            auto             eq   = tok.find('=');
            std::string_view name = tok.substr(2, eq == std::string_view::npos ? tok.size() - 2 : eq - 2);
            OptionBase      *o    = detail::find_long(name);
            if (!o) {
                if (unknown_args)
                    unknown_args->push_back(std::string(tok));
                continue;
            }

            std::optional<std::string_view> val;
            if (eq != std::string_view::npos) {
                val = tok.substr(eq + 1);
            } else if (o->value_expected == ValueExpected::ValueRequired) {
                // Look ahead; only consume if it doesn't look like another option
                if (i + 1 < args.size()) {
                    std::string_view next = args[i + 1];
                    if (!looks_like_option_token(next)) {
                        val = std::string_view(args[++i]); // consume as value
                    } // else leave val = nullopt to allow ImplicitValue(...)
                } // else leave val = nullopt
            }

            std::string err;
            if (!o->parse_token(name, val, err)) {
                fmt::print(stderr, "error: {}\n", err);
                return UNKNOWN_ARGUMENT;
            }
            if (o->on_seen)
                o->on_seen();
            if (o->long_name == "help") {
                print_help(prog);
                return HELP;
            }
            if (o->long_name == "version") {
                print_version(prog, version);
                return VERSION;
            }
            continue;
        }

        // Short options (possibly bundled): -abc, -o value, -ovalue
        if (tok.size() >= 2 && tok[0] == '-') {
            for (size_t j = 1; j < tok.size(); ++j) {
                char        c = tok[j];
                OptionBase *o = detail::find_short(c);
                if (!o) {
                    if (unknown_args)
                        unknown_args->push_back(fmt::format("-{}", c));
                    continue;
                }

                std::optional<std::string_view> val;
                bool                            last_in_bundle = (j + 1 == tok.size());
                if (o->value_expected == ValueExpected::ValueRequired) {
                    if (!last_in_bundle) {
                        // remainder of bundle is the value: -ovalue
                        val = tok.substr(j + 1);
                        j   = tok.size();
                    } else {
                        // last in bundle; optionally consume next token if it's a value
                        if (i + 1 < args.size()) {
                            std::string_view next = args[i + 1];
                            if (!looks_like_option_token(next)) {
                                val = std::string_view(args[++i]); // consume as value
                            } // else leave nullopt to allow ImplicitValue(...)
                        } // else leave nullopt
                    }
                }

                std::string err;
                if (!o->parse_token(std::string_view(&c, 1), val, err)) {
                    fmt::print(stderr, "error: {}\n", err);
                    return UNKNOWN_ARGUMENT;
                }
                if (o->on_seen)
                    o->on_seen();
                if (o->long_name == "help") {
                    print_help(prog);
                    return HELP;
                }
                if (o->long_name == "version") {
                    print_version(prog, version);
                    return VERSION;
                }
            }
            continue;
        }

        // Bare token -> positional or unknown
        std::string err;
        if (!consume_positional(tok, err)) {
            fmt::print(stderr, "error: {}\n", err);
            return UNKNOWN_ARGUMENT;
        }
    }

    // Validate required/occurrence
    for (auto *o : Registry::instance().options) {
        if ((o->occurrence == Occurrence::Required || o->occurrence == Occurrence::OneOrMore) && o->occurrences == 0) {
            fmt::print(stderr, "error: missing required option '--{}'\n", o->long_name);
            return MISSING_REQUIRED;
        }
        std::string err;
        if (!o->validate(err)) {
            fmt::print(stderr, "error: {}\n", err);
            return INVALID_ARGUMENT;
        }
    }

    // Validate exclusions.
    for (auto *exc : Registry::instance().exclusions) {
        if (!exc->verify_exclusions()) {
            auto found = exc->found_options();

            auto it = found.begin();

            fmt::print(stderr, "error: incompatible arguments found: ");

            for (int i = 0; i < found.size(); i++, it++) {
                if (i != found.size() - 1) {
                    fmt::print(stderr, "--{}, ", (*it)->long_name);
                } else {
                    fmt::print(stderr, "--{}\n", (*it)->long_name);
                }
            }
            return INCOMPATIBLE_ARGUMENT;
        }
    }

    return SUCCESS;
}

}
}
