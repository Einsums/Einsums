//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------


#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

namespace einsums::cl {

bool parse_value(std::string_view value, std::string &out, std::string &) {
    out.assign(value.begin(), value.end());
    return true;
}

bool parse_value(std::string_view value, bool &out, std::string &err) {
    if (value == "1" || value == "true" || value == "on" || value == "yes") {
        out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "off" || value == "no") {
        out = false;
        return true;
    }
    err = fmt::format("invalid boolean '{}', expected true/false", value);
    return false;
}

bool parse_value(std::string_view value, int &out, std::string &err) {
    int   out_val{};
    auto [pointer, errorcode] = std::from_chars(value.cbegin(), value.cend(), out_val);
    if (errorcode == std::errc{} && pointer == value.cend()) {
        out = out_val;
        return true;
    }
    err = fmt::format("invalid integer '{}'", value);
    return false;
}

bool parse_value(std::string_view sv, long &out, std::string &err) {
    long int   out_val{};
    auto [pointer, errorcode] = std::from_chars(value.cbegin(), value.cend(), out_val);
    if (errorcode == std::errc{} && pointer == value.cend()) {
        out = out_val;
        return true;
    }
    err = fmt::format("invalid integer '{}'", value);
    return false;
}

bool parse_value(std::string_view sv, long long &out, std::string &err) {
    long long int   out_val{};
    auto [pointer, errorcode] = std::from_chars(value.cbegin(), value.cend(), out_val);
    if (errorcode == std::errc{} && pointer == value.cend()) {
        out = out_val;
        return true;
    }
    err = fmt::format("invalid integer '{}'", value);
    return false;
}

bool parse_value(std::string_view sv, double &out, std::string &err) {
    try {
        size_t      idx = 0;
        std::string tmp(sv);
        double      v = std::stod(tmp, &idx);
        if (idx == tmp.size()) {
            out = v;
            return true;
        }
    } catch (std::invalid_argument) {
        err = fmt::format("invalid real '{}'", sv);
        return false;
    } catch (std::out_of_range) {
        err = fmt::format("real value '{}' out of range!", sv);
        return false;
    }
}

std::shared_ptr<ExclusiveCategory> make_yes_no(Flag &yes_flag, Flag &no_flag, bool default_value) {
    auto out = std::make_shared<ExclusiveCategory>();

    yes_flag.exclusions = out.get();
    no_flag.exclusions = out.get();

    out->options.push_back(&yes_flag);
    out->options.push_back(&no_flag);

    yes_flag.set_on_unseen = false;
    no_flag.set_on_unseen = false;

    yes_flag.value = true;
    no_flag.value = false;

    if (default_value) {
        if (yes_flag.bound)
            *yes_flag.bound = true;
        if (yes_flag.setter)
            yes_flag.setter(true);
    } else {
        if (no_flag.bound)
            *no_flag.bound = true;
        if (no_flag.setter)
            no_flag.setter(true);
    }

    return out;
}

std::map<std::string, std::string, std::less<>> read_config(std::string_view path) {
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
        size_t start = 0;
        while (start <= buf.size()) {
            size_t      end  = buf.find_first_of("\r\n", start);
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

    // Minimal flat JSON object: { "k": value }
    size_t i    = 0;
    auto   s    = buf;
    auto   skip = [&] {
        while (i < s.size() && std::isspace((unsigned char)s[i]))
            ++i;
    };
    i = 0;
    skip();
    if (i >= s.size() || s[i] != '{')
        return kv;
    ++i;
    while (true) {
        skip();
        if (i < s.size() && s[i] == '}') {
            ++i;
            break;
        }
        if (i >= s.size() || s[i] != '\"')
            break;
        ++i;
        std::string key;
        while (i < s.size() && s[i] != '\"') {
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
        ++i;
        skip();
        std::string val;
        if (i < s.size() && s[i] == '\"') {
            ++i;
            while (i < s.size() && s[i] != '\"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    val.push_back(s[i + 1]);
                    i += 2;
                } else {
                    val.push_back(s[i++]);
                }
            }
            if (i < s.size())
                ++i;
        } else if (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '-' || s[i] == '+')) {
            size_t j = i;
            while (j < s.size() &&
                   (std::isdigit((unsigned char)s[j]) || s[j] == '.' || s[j] == 'e' || s[j] == 'E' || s[j] == '+' || s[j] == '-'))
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
            lk.push_back((char)std::tolower((unsigned char)c));
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

void print_version(std::string_view prog, std::string_view ver) {
    if (!ver.empty())
        fmt::print("{} {}\n", prog, ver);
}

} // namespace einsums::cl
