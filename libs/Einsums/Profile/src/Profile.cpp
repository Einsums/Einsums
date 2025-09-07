//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/Profile.hpp>

#if defined(EINSUMS_HAVE_PROFILER)
namespace einsums::profile {

namespace {

auto strip_ansi_sequences(std::string const &s) -> std::string {
    std::string out;
    out.reserve(s.size());

    for (size_t i = 0; i < s.size();) {
        unsigned char const c = s[i];
        if (c == '\x1b') { // ESC
            if (i + 1 >= s.size()) {
                ++i;
                break;
            }

            unsigned char const c1 = s[i + 1];

            if (c1 == '[') {
                // CSI: ESC [ ... final byte in @-~
                i += 2;
                while (i < s.size()) {
                    unsigned char const cc = s[i++];
                    if (cc >= '@' && cc <= '~')
                        break; // final byte
                }
                continue;
            } else if (c1 == ']') {
                // OSC: ESC ] ... terminated by BEL or ESC \
                i += 2;
                while (i < s.size()) {
                    if (s[i] == '\x07') {
                        ++i;
                        break;
                    } // BEL
                    if (s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '\\') {
                        i += 2;
                        break;
                    } // ESC \
                    if (s[i] == '\x1b') break; // new ESC -> bail out
                    ++i;
                }
                continue;
            } else {
                // Other ESC sequences (2-byte minimum)
                i += 2;
                continue;
            }
        } else {
            out.push_back(static_cast<char>(c));
            ++i;
        }
    }
    return out;
}

// compute visible width (in code units; doesn't attempt grapheme width)
auto visible_width(std::string const &s) -> size_t {
    return strip_ansi_sequences(s).size();
}

// print the original string (keeps ANSI/hyperlink escapes) and pad based on visible width
void fmt_pad_and_print(std::ostream &os, std::string const &s, size_t width) {
    size_t const vlen = visible_width(s);
    fmt::print(os, "{}", s);
    if (vlen < width) {
        fmt::print(os, "{}", std::string(width - vlen, ' '));
    }
}

template <typename... Args>
void safe_print(std::ostream &os, fmt::text_style style, std::string_view format_str, Args &&...args) {
    std::string output;
    if (detail::is_terminal(os)) {
        output = fmt::format(style, fmt::runtime(format_str), std::forward<Args>(args)...);
    } else {
        output = fmt::format(fmt::runtime(format_str), std::forward<Args>(args)...);
    }
    fmt::print(os, "{}", output);
}

// ---------------------- Helpers for OSC-8 clickable links ----------------------
auto make_clickable_file_line(std::string const &file, int line, std::string const &display) -> std::string {
    if (file.empty() || line <= 0)
        return display;
    // OSC 8 format: ESC ] 8 ; ; URI ST <text> ESC ] 8 ; ; ST
    // ST is ESC '\'
    std::string const uri = "file://" + file + ":" + std::to_string(line);
    std::string const esc = "\x1b]8;;";
    std::string const st  = "\x1b\\";
    return esc + uri + st + display + esc + st;
}

// ------------------ printing helpers ------------------
auto ns_to_ms(ns const t) -> double {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t).count();
}

auto escape_json(std::string const &s) -> std::string {
    std::ostringstream o;
    for (unsigned char const c : s) { // unsigned to avoid sign-extension issues
        switch (c) {
        case '"':
            o << "\\\"";
            break;
        case '\\':
            o << "\\\\";
            break;
        case '\b':
            o << "\\b";
            break;
        case '\f':
            o << "\\f";
            break;
        case '\n':
            o << "\\n";
            break;
        case '\r':
            o << "\\r";
            break;
        case '\t':
            o << "\\t";
            break;
        default:
            if (c <= 0x1F) { // control characters
                o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec << std::setfill(' ');
            } else {
                o << c;
            }
        }
    }
    return o.str();
}

// color selection matching PNG thresholds
auto style_for_percent(double p) -> fmt::text_style {
    // thresholds: >80 hot red (bold), >65 light red, >20 yellow, <20 cyan/dim gray
    if (p > 80.0)
        return fg(fmt::color::red) | fmt::emphasis::bold;
    if (p > 65.0)
        return fg(fmt::color{0xFF6B6B}) | fmt::emphasis::bold; // light red
    if (p > 20.0)
        return fg(fmt::color::yellow);
    if (p > 0.0)
        return fg(fmt::color::turquoise);
    return fg(fmt::color::gray);
}

} // namespace

void Profiler::print(bool detailed, std::ostream &os) {
    std::lock_guard const lock(_mutex);
    // compute per-thread totals (sum of top-level nodes exclusive)
    for (auto &tkv : thread_data()) {
        auto const &thread          = tkv.first;
        double      thread_total_ms = 0.0;
        for (auto &c : tkv.second.children) {
            thread_total_ms += ns_to_ms(c.second->total_exclusive);
        }

        // header
        fmt::print(os, "\n");
        safe_print(os, fmt::emphasis::bold | fg(fmt::color::white), "Thread: {}  (total exclusive: {:-7.3f} ms)\n", thread,
                   thread_total_ms);
        fmt::print(os, "{:-^142}\n", "");

        // column header: % | time(ms) | name | file:line | func
        if (!detailed) {
            fmt::print(os, " {:>10}  {:<10}  {:<60}  {:<30}  {:<}\n", "time(ms)", "count", "name", "file:line", "function");
            fmt::print(os, "{:-^142}\n", "");
        } else {
            // show extra cols for min/max/avg and counters
            fmt::print(os, " {:>10}  {:<60}  {:<30}  {:<20}  {:>8} {:>8} {:>8}\n", "time(ms)", "name", "file:line", "function", "min",
                       "max", "avg");
            fmt::print(os, "{:-^120}\n", "");
        }

        std::vector<AggNode *> nodes;
        for (auto &c : tkv.second.children)
            nodes.push_back(c.second.get());

        for (auto *n : nodes) {
            print_node_recursive(os, n, thread_total_ms, 0, detailed);
        }
        fmt::print(os, "\n");
    }
}

auto Profiler::export_json(std::string const &path) -> std::optional<std::string> {
    std::lock_guard const lock(_mutex);
    std::ofstream         ofs(path, std::ios::trunc);
    if (!ofs)
        return std::nullopt;
    ofs << "{\n";
    bool firstThread = true;
    for (auto &tkv : thread_data()) {
        if (!firstThread)
            ofs << ",\n";
        firstThread = false;
        ofs << fmt::format("  \"{}\": ", tkv.first);
        write_node_json(ofs, tkv.second, 2);
    }
    ofs << "\n}\n";
    return path;
}

void Profiler::write_node_json(std::ostream &ofs, AggNode const &n, int indent) {
    std::string const ind(indent, ' ');
    ofs << ind << "{\n";
    ofs << ind << "  \"name\": \"" << escape_json(n.name) << "\",\n";
    ofs << ind << "  \"call_count\": " << n.call_count << ",\n";
    ofs << ind << "  \"exclusive_ms\": " << std::fixed << std::setprecision(6) << ns_to_ms(n.total_exclusive) << ",\n";
    ofs << ind << "  \"exclusive_min_ms\": " << ns_to_ms(n.exclusive_min) << ",\n";
    ofs << ind << "  \"exclusive_max_ms\": " << ns_to_ms(n.exclusive_max) << ",\n";

    // counters
    ofs << ind << "  \"counters\": {";
    bool first = true;
    for (auto &c : n.counters_total) {
        if (!first)
            ofs << ", ";
        first              = false;
        uint64_t const tot = c.second;
        uint64_t const mn  = n.counters_min.at(c.first);
        uint64_t const mx  = n.counters_max.at(c.first);
        ofs << "\"" << escape_json(c.first) << R"(": {"total": )" << tot << ", \"min\": " << mn << ", \"max\": " << mx << "}";
    }
    ofs << "},\n";

    ofs << ind << "  \"children\": [\n";
    bool firstChild = true;
    for (auto &ch : n.children) {
        if (!firstChild)
            ofs << ",\n";
        firstChild = false;
        write_node_json(ofs, *ch.second, indent + 4);
    }
    ofs << "\n" << ind << "  ]\n";
    ofs << ind << "}";
}

void Profiler::print_node_recursive(std::ostream &os, AggNode const *n, double thread_total_ms, int depth, bool detailed) {
    // depth indentation
    std::string const indent(2 * depth, ' ');
    double            excl_ms = ns_to_ms(n->total_exclusive);
    double const      pct     = (thread_total_ms > 0.0) ? (excl_ms / thread_total_ms * 100.0) : 0.0;

    // name truncated to fit
    std::string name = indent + n->name;
    if (name.size() > 60)
        name = name.substr(0, 57) + "...";

    // file:line & func retrieval is not stored in AggNode (we store in ActiveFrame). For this summary,
    // we will not show file:line per node unless we had stored it per node earlier.
    // The user requested clickable file:line; to provide that we need to attach one canonical file:line to the aggregated node.
    // For simplicity, we will display the first-seen file:line if present. (We can extend to store file+line per node.)
    // We'll attempt to extract a stored file:line from a node metadata map if available (not implemented) — fallback to blank.
    std::string func = n->function;

    // style
    auto style = style_for_percent(pct);

    // time, name, file:line (clickable), func
    fmt::print(os, " ");
    fmt::print(os, "{:10.3f}  ", excl_ms);
    fmt::print(os, "{:10}  ", n->call_count);
    fmt::print(os, "{:<60}  ", name);
    // clickable file:line (if present)
    // inside print_node_recursive when printing one node:
    if (!n->file.empty()) {
        // show just filename:line as the visible text, but link to full path
        std::string shortname;
        try {
            shortname = std::filesystem::path(n->file).filename().string();
        } catch (...) {
            shortname = n->file; // fallback
        }
        std::string const file_display = fmt::format("{}:{}", shortname, n->line);
        std::string       clickable;
        if (detail::is_terminal(os)) {
            clickable = make_clickable_file_line(n->file, n->line, file_display);
        } else {
            clickable = file_display;
        }
        fmt_pad_and_print(os, clickable, 30);
        fmt::print(os, "  "); // extra spacing between columns if needed
    } else {
        fmt::print(os, "{:<30}  ", ""); // no file info
    }

    // function name
    fmt::print(os, "{:<}\n", func);

    if (detailed) {
        double min_ms = ns_to_ms(n->exclusive_min);
        double max_ms = ns_to_ms(n->exclusive_max);
        double avg_ms = (n->call_count > 0) ? (ns_to_ms(n->total_exclusive) / n->call_count) : 0.0;
        fmt::print(os, "{:6}   {:>10.3f}  (min {:>6.3f}  max {:>6.3f}  avg {:>6.3f})\n", "", excl_ms, min_ms, max_ms, avg_ms);
        if (!n->counters_total.empty()) {
            // print counters in one compact line
            fmt::print(os, "{:6}   Counters:", "");
            for (auto &c : n->counters_total) {
                uint64_t tot = c.second;
                uint64_t mn  = n->counters_min.at(c.first);
                uint64_t mx  = n->counters_max.at(c.first);
                double   avg = (n->call_count > 0) ? static_cast<double>(tot) / n->call_count : 0.0;
                fmt::print(os, " {}(tot={},min={},max={},avg={:.1f})", c.first, tot, mn, mx, avg);
            }
            fmt::print(os, "\n");
        }
    }

    std::vector<AggNode const *> children;
    for (auto &c : n->children)
        children.push_back(c.second.get());

    for (auto *ch : children) {
        print_node_recursive(os, ch, thread_total_ms, depth + 1, detailed);
    }
}

} // namespace einsums::profile
#endif
