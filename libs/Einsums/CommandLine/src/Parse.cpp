//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/Compat/FmtAlternatives.hpp>
#include <Einsums/Compat/PosixAlternatives.hpp>
#include <Einsums/StringUtil/StringOps.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define BUFFER_SIZE 4096

namespace einsums::cl {

namespace {

bool looks_json(std::string const &buf) {
    for (char ch : buf) {
        if (ch == '{') {
            return true;
        } else if (!std::isspace(ch)) {
            return false;
        }
    }

    std::fprintf(stderr, "Warning: config file is empty.\n");
    return false;
}

size_t skip(std::string const &s, size_t i) {
    size_t out = i;
    while (out < s.size() && std::isspace(s[out])) {
        out++;
    }
    return out;
}

std::string get_file(std::FILE *fp) {
    char buff[BUFFER_SIZE];

    std::ostringstream out;

    do {
        auto res = std::fgets(buff, BUFFER_SIZE, fp);

        if (res == buff) {
            out << buff;
        }
    } while (!std::feof(fp) && !std::ferror(fp));

    return out.str();
}

bool looks_like_option_token(std::string_view sv) {
    if (sv.size() >= 1 && sv[0] == '-') {
        // Treat numeric-looking tokens like "-5" or "-3.14" as values, not options
        if (sv.size() >= 2 && std::isdigit(static_cast<unsigned char>(sv[1])))
            return false;
        return true;
    }
    return false;
}

bool consume_positional(std::string_view token, size_t *pos_index, std::vector<std::string> *unknown_args, std::vector<OptionBase *> &pos) {
    if (*pos_index >= pos.size()) {
        // No positional to consume -> treat as unknown (per your policy)
        if (unknown_args) {
            unknown_args->emplace_back(token);
        }
        return true;
    }

    OptionBase *p  = pos[*pos_index];
    bool        ok = p->parse_token(p->get_long_name(), token, "error: ");
    if (!ok) {
        return false;
    }

    p->seen_on_cli();

    // Stay on the same positional if it's a List<std::string>
    // so it can keep capturing subsequent tokens.
    if (dynamic_cast<List<std::string> *>(p) == nullptr) {
        (*pos_index)++;
    }

    return true;
}

} // namespace

std::map<std::string, std::string, std::less<>> read_config(std::string_view const &path) {
    std::map<std::string, std::string, std::less<>> kv;
    if (path.empty()) {
        return kv;
    }
    std::FILE *f = std::fopen(std::string(path).c_str(), "r");
    if (!f) {
        std::perror("Could not open file!\n");
        return kv;
    }

    auto buf = get_file(f);

    std::fclose(f);

    if (!looks_json(buf)) {
        size_t start = 0;
        while (start < buf.size()) {

            size_t      end  = buf.find_first_of("\r\n", start);
            std::string line = (end == std::string::npos) ? buf.substr(start) : buf.substr(start, end - start);
            start            = (end == std::string::npos) ? buf.size() + 1 : end + 1;
            if (line.empty() || line[0] == '#') {
                continue;
            }
            auto eq = line.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            std::string k = string_util::trim(line.substr(0, eq)), v = string_util::trim(line.substr(eq + 1));
            kv[string_util::lower(k)] = v;
        }
        return kv;
    }

    // Minimal flat JSON object: { "k": value }
    size_t i = 0;
    i        = skip(buf, 0);
    if (i >= buf.size() || buf[i] != '{') {
        return kv;
    }
    ++i;
    while (i < buf.size()) {
        i = skip(buf, i);
        if (i < buf.size() && buf[i] == '}') {
            ++i;
            break;
        }
        if (i >= buf.size() || buf[i] != '\"') {
            break;
        }
        ++i;
        std::string key;
        while (i < buf.size() && buf[i] != '\"') {
            if (buf[i] == '\\' && i + 1 < buf.size()) {
                key.push_back(buf[i + 1]);
                i += 2;
            } else {
                key.push_back(buf[i++]);
            }
        }
        if (i < buf.size()) {
            ++i;
        }
        i = skip(buf, i);
        if (i >= buf.size() || buf[i] != ':') {
            break;
        }
        ++i;
        i = skip(buf, i);
        std::string val;
        if (i < buf.size() && buf[i] == '\"') {
            ++i;
            while (i < buf.size() && buf[i] != '\"') {
                if (buf[i] == '\\' && i + 1 < buf.size()) {
                    val.push_back(buf[i + 1]);
                    i += 2;
                } else {
                    val.push_back(buf[i++]);
                }
            }
            if (i < buf.size())
                ++i;
        } else if (i < buf.size() && (std::isdigit((unsigned char)buf[i]) || buf[i] == '-' || buf[i] == '+')) {
            size_t j = i;
            while (j < buf.size() && (std::isdigit((unsigned char)buf[j]) || buf[j] == '.' || buf[j] == 'e' || buf[j] == 'E' ||
                                      buf[j] == '+' || buf[j] == '-')) {
                ++j;
            }
            val = buf.substr(i, j - i);
            i   = j;
        } else if (buf.compare(i, 4, "true") == 0) {
            val = "true";
            i += 4;
        } else if (buf.compare(i, 5, "false") == 0) {
            val = "false";
            i += 5;
        } else {
            while (i < buf.size() && buf[i] != ',' && buf[i] != '}') {
                ++i;
            }
        }
        std::string lk;
        for (char c : key) {
            lk.push_back((char)std::tolower((unsigned char)c));
        }
        kv[lk] = val;
        i      = skip(buf, i);
        if (i < buf.size() && buf[i] == ',') {
            ++i;
            continue;
        }
        i = skip(buf, i);
        if (i < buf.size() && buf[i] == '}') {
            ++i;
            break;
        }
    }
    return kv;
}

ParseResult parse_internal(std::vector<std::string> const &args, char const *programName, std::string_view version,
                           std::map<std::string, std::string, std::less<>> const *config, std::vector<std::string> *unknown_args) {
    Builtins                &builtins = Builtins::get_singleton();
    GlobalConfigMapLockScope __;
    std::string              prog = programName ? programName : (!args.empty() ? args[0] : "Einsums");

    for (auto *o : Registry::get_singleton().get_options()) {
        o->finalize_default();
    }

    // Apply config first (defaults < config < CLI)
    if (config && !config->empty()) {
        for (auto *o : Registry::get_singleton().get_options()) {
            if (o->is_positional()) {
                continue;
            }
            auto it = config->find(o->get_long_name());
            if (it == config->end()) {
                continue;
            }

            std::optional<std::string_view> v;
            if (!it->second.empty()) {
                v = std::string_view(it->second);
            }
            if (!o->parse_token(o->get_long_name(), v, einsums::format("config error for {}: ", o->get_long_name()),
                                /*from_config=*/true)) {
                return {.ok = false, .exit_code = 1};
            }
        }
    }

    size_t pos_index = 0;

    // Parse CLI
    for (size_t i = 1; i < args.size(); ++i) {
        std::string_view tok(args[i]);

        // Everything after "--" -> unknown_args
        if (tok == "--") {
            i++;
            while (i < args.size()) {
                if (unknown_args) {
                    unknown_args->push_back(args[i]);
                }
                i++;
            }
            break;
        }

        // Long options: --name or --name=value
        if (tok.size() >= 2 && tok[0] == '-' && tok[1] == '-') {
            auto             eq   = tok.find('=');
            std::string_view name = tok.substr(2, eq == std::string_view::npos ? tok.size() - 2 : eq - 2);
            OptionBase      *o    = detail::find_long(name);
            if (!o) {
                if (unknown_args) {
                    unknown_args->emplace_back(tok);
                }
                continue;
            }

            std::optional<std::string_view> val;
            if (eq != std::string_view::npos) {
                val = tok.substr(eq + 1);
            } else if (o->get_value_expected() == ValueExpected::ValueRequired) {
                // Look ahead; only consume if it doesn't look like another option
                if (i + 1 < args.size()) {
                    std::string_view next = args[i + 1];
                    if (!looks_like_option_token(next)) {
                        val = std::string_view(args[++i]); // consume as value
                    } // else leave val = nullopt to allow ImplicitValue(...)
                } // else leave val = nullopt
            }

            if (!o->parse_token(name, val, "error: ")) {
                return {.ok = false, .exit_code = 1};
            }
            if (o->get_impl()->on_seen_) {
                o->get_impl()->on_seen_();
            }
            if (o->get_long_name() == "help") {
                print_help(prog);
                return {.ok = false, .exit_code = 0};
            }
            if (o->get_long_name() == "version") {
                print_version(prog, version);
                return {.ok = false, .exit_code = 0};
            }
            continue;
        }

        // Short options (possibly bundled): -abc, -o value, -ovalue
        if (tok.size() >= 2 && tok[0] == '-') {
            for (size_t j = 1; j < tok.size(); ++j) {
                char        c = tok[j];
                OptionBase *o = detail::find_short(c);
                if (!o) {
                    if (unknown_args) {
                        unknown_args->push_back(einsums::format("-{}", c));
                    }
                    continue;
                }

                std::optional<std::string_view> val;
                bool                            last_in_bundle = (j + 1 == tok.size());
                if (o->get_value_expected() == ValueExpected::ValueRequired) {
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

                if (!o->parse_token(std::string_view(&c, 1), val, "error: ")) {
                    return {.ok = false, .exit_code = 1};
                }
                //                if (o->get_impl()->on_seen_) {
                //                    o->get_impl()->on_seen_();
                //                }
                if (o->get_long_name() == "help") {
                    print_help(prog);
                    return {.ok = false, .exit_code = 0};
                }
                if (o->get_long_name() == "version") {
                    print_version(prog, version);
                    return {.ok = false, .exit_code = 0};
                }
            }
            continue;
        }

        auto pos = detail::positional_options();

        // Bare token -> positional or unknown
        if (!consume_positional(tok, &pos_index, unknown_args, pos)) {
            return {.ok = false, .exit_code = 1};
        }
    }

    // Validate required/occurrence
    for (auto *o : Registry::get_singleton().get_options()) {
        if ((o->get_occurrence() == Occurrence::Required || o->get_occurrence() == Occurrence::OneOrMore) && o->num_occurrences() == 0) {
            fmt::print(stderr, "error: missing required option '--{}'\n", o->get_long_name());
            return {.ok = false, .exit_code = 1};
        }
        if (!o->validate("error: ")) {
            return {.ok = false, .exit_code = 1};
        }
    }

    // Validate exclusions.
    for (auto const *exc : Registry::get_singleton().get_exclusions()) {
        if (!exc->verify_exclusions()) {
            auto found = exc->found_options();

            auto it = found.begin();

            std::fprintf(stderr, "error: incompatible arguments found: \n");

            for (int i = 0; i < found.size(); i++, it++) {
                fmt::print(stderr, "--{}\n", (*it)->get_long_name());
            }
            return {.ok = false, .exit_code = 1};
        }
    }

    return {.ok = true, .exit_code = 0};
}
} // namespace einsums::cl