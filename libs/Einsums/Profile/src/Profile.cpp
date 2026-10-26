//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Compat/FmtAlternatives.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/Profile.hpp>

namespace einsums::profile {

namespace {

size_t visible_width(std::string const &s) {
    size_t out = 0;

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
                    if (cc >= '@' && cc <= '~') {
                        break; // final byte
                    }
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
            out++;
            ++i;
        }
    }
    return out;
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
        output = einsums::format(style, fmt::runtime(format_str), std::forward<Args>(args)...);
    } else {
        output = einsums::format(fmt::runtime(format_str), std::forward<Args>(args)...);
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

double variance(uint64_t n, int64_t M2) {
    return (n > 1) ? double(M2) / (n - 1) : 0.0;
};
double stddev(uint64_t n, int64_t M2) {
    return sqrt(variance(n, M2));
};

} // namespace

EINSUMS_SINGLETON_IMPL(Profiler)

Profiler::~Profiler() {
    std::fprintf(stderr, "Finalizing Profiler.\n");
}

void Profiler::push(std::string const &name, std::string const &file, int line, std::string const &func) {
    auto now = Clock::now();

#ifdef EINSUMS_HAVE_TRACY
    // dynamic runtime zone name using ScopedZone
    TracyStack::get_multiton().emplace(line, file.c_str(), file.size(), func.c_str(), func.size(), name.c_str(), name.size(), 1);
#endif

    ActiveStack::get_multiton().emplace(name, now, file, line, func);
}

void Profiler::pop() {
    auto  now          = Clock::now();
    auto &active_stack = ActiveStack::get_multiton();
    if (active_stack.empty()) {
        return;
    }
    ActiveFrame const frame = active_stack.top();
    active_stack.pop();

    ns const duration  = std::chrono::duration_cast<ns>(now - frame.start());
    ns const exclusive = duration - frame.child_time();

    // build path from root to this node
    std::vector<std::string> path;
    for (auto &f : active_stack) {
        path.push_back(f.name());
    }
    path.push_back(frame.name());

    // collect counter deltas (if any)
    //        std::map<std::string, uint64_t> const deltas;

#ifdef EINSUMS_HAVE_TRACY
    // pop the tracy zone
    if (!TracyStack::get_multiton().empty()) {
        TracyStack::get_multiton().pop_back();
    }
#endif

    // update aggregated data
    {
        std::lock_guard const lock(_mutex);
        auto                 &root = thread_data()[thread_key()];
        AggNode              *cur  = std::addressof(root);
        for (auto &p : path) {
            cur = cur->find_or_add(p).get();
        }

        cur->set_file(frame.file());
        cur->set_line(frame.line());
        cur->set_function(frame.function());

        cur->inc_call_count();

        cur->inc_exclusive(exclusive);

        // merge counters
        //            for (auto &kv : deltas) {
        //                std::string const &ename = kv.first;
        //                uint64_t const     val   = kv.second;
        //                cur->counters_total[ename] += val;
        //                auto itmin = cur->counters_min.find(ename);
        //                if (itmin == cur->counters_min.end()) {
        //                    cur->counters_min[ename] = val;
        //                    cur->counters_max[ename] = val;
        //                } else {
        //                    if (val < cur->counters_min[ename])
        //                        cur->counters_min[ename] = val;
        //                    if (val > cur->counters_max[ename])
        //                        cur->counters_max[ename] = val;
        //                }
        //            }
    }

    // add duration to parent's child_time so parent exclusive deducts it
    if (!active_stack.empty()) {
        active_stack.top().child_time() += duration;
    }
}

void Profiler::print(bool detailed, std::ostream &os) {
    std::lock_guard const lock(_mutex);
    // compute per-thread totals (sum of top-level nodes exclusive)
    for (auto &tkv : thread_data()) {
        auto const &thread          = tkv.first;
        double      thread_total_ms = 0.0;
        for (auto const &c : tkv.second.children()) {
            thread_total_ms += ns_to_ms(c.second->total_exclusive());
        }

        // header
        fmt::print(os, "\n");
        safe_print(os, fmt::emphasis::bold | fg(fmt::color::white), "Thread: {}  (total exclusive: {:-7.3f} ms)\n", thread,
                   thread_total_ms);
        fmt::print(os, "{:-^157}\n", "");

        // column header: % | time(ms) | name | file:line | func
        if (!detailed) {
            fmt::print(os, " {:>10}  {:^10}  {:^13}  {:<60}  {:<30}  {:<}\n", "total(ms)", "count", "mean(ms)", "name", "file:line",
                       "function");
            fmt::print(os, "{:-^157}\n", "");
        } else {
            // show extra cols for min/max/avg and counters
            fmt::print(os, " {:>10}  {:<60}  {:<30}  {:<20}  {:>8} {:>8} {:>8}\n", "total(ms)", "name", "file:line", "function", "min",
                       "max", "avg");
            fmt::print(os, "{:-^120}\n", "");
        }

        std::list<AggNode const *> nodes;
        for (auto const &c : tkv.second.children()) {
            nodes.push_back(c.second.get());
        }

        for (auto const *n : nodes) {
            print_node_recursive(os, n, thread_total_ms, 0, detailed);
        }
        fmt::print(os, "\n");
    }
}

bool Profiler::export_json(std::string const &path) {
    std::lock_guard const lock(_mutex);
    std::ofstream         ofs(path, std::ios::trunc);
    if (!ofs) {
        return false;
    }
    ofs << "{\n";
    bool first_thread = true;
    for (auto &tkv : thread_data()) {
        if (!first_thread) {
            ofs << ",\n";
        }
        first_thread = false;
        ofs << einsums::format("  \"{}\": ", tkv.first);
        write_node_json(ofs, tkv.second, 2);
    }
    ofs << "\n}\n";
    return true;
}

void Profiler::write_node_json(std::ostream &ofs, AggNode const &n, int indent) {
    std::string const ind(indent, ' ');
    ofs << ind << "{\n";
    ofs << ind << "  \"name\": \"" << escape_json(n.name()) << "\",\n";
    ofs << ind << "  \"call_count\": " << n.call_count() << ",\n";
    ofs << ind << "  \"exclusive_ms\": " << std::fixed << std::setprecision(6) << ns_to_ms(n.total_exclusive()) << ",\n";
    ofs << ind << "  \"exclusive_min_ms\": " << ns_to_ms(n.exclusive_min()) << ",\n";
    ofs << ind << "  \"exclusive_max_ms\": " << ns_to_ms(n.exclusive_max()) << ",\n";

    // counters
    ofs << ind << "  \"counters\": {";
    bool first = true;
    for (auto const &c : n.counters_total()) {
        if (!first) {
            ofs << ", ";
        }
        first              = false;
        uint64_t const tot = c.second;
        uint64_t const mn  = n.counters_min().at(c.first);
        uint64_t const mx  = n.counters_max().at(c.first);
        ofs << "\"" << escape_json(c.first) << R"(": {"total": )" << tot << ", \"min\": " << mn << ", \"max\": " << mx << "}";
    }
    ofs << "},\n";

    ofs << ind << "  \"children\": [\n";
    bool first_child = true;
    for (auto const &ch : n.children()) {
        if (!first_child) {
            ofs << ",\n";
        }
        first_child = false;
        write_node_json(ofs, *ch.second, indent + 4);
    }
    ofs << "\n" << ind << "  ]\n";
    ofs << ind << "}";
}

void Profiler::print_node_recursive(std::ostream &os, AggNode const *n, double thread_total_ms, int depth, bool detailed) {
    // depth indentation
    std::string const indent(2 * depth, ' ');
    double            excl_ms = ns_to_ms(n->total_exclusive());
    double const      pct     = (thread_total_ms > 0.0) ? (excl_ms / thread_total_ms * 100.0) : 0.0;

    // name truncated to fit
    std::string name = indent + n->name();
    if (name.size() > 60) {
        name = name.substr(0, 57) + "...";
    }

    // file:line & func retrieval is not stored in AggNode (we store in ActiveFrame). For this summary,
    // we will not show file:line per node unless we had stored it per node earlier.
    // The user requested clickable file:line; to provide that we need to attach one canonical file:line to the aggregated node.
    // For simplicity, we will display the first-seen file:line if present. (We can extend to store file+line per node.)
    // We'll attempt to extract a stored file:line from a node metadata map if available (not implemented) — fallback to blank.
    std::string func = n->function();

    // style
    auto style = style_for_percent(pct);

    // time, name, file:line (clickable), func
    fmt::print(os, " ");
    fmt::print(os, "{:10.3f}  ", excl_ms);
    fmt::print(os, "{:10}  ", n->call_count());

    // average and standard deviation
    std::ostringstream oss;
    fmt::print(oss, fmt::runtime("{:7.3f}±{:3.3f}"), double(n->total_exclusive_mean()) / 1'000'000.0,
               stddev(n->call_count(), n->total_exclusive_M2()) / 1'000'000.0);
    fmt::print(os, "{:13}  ", oss.str());

    fmt::print(os, "{:<60}  ", name);
    // clickable file:line (if present)
    // inside print_node_recursive when printing one node:
    if (!n->file().empty()) {
        // show just filename:line as the visible text, but link to full path
        std::string shortname;
        try {
            shortname = std::filesystem::path(n->file()).filename().string();
        } catch (...) {
            shortname = n->file(); // fallback
        }
        std::string const file_display = einsums::format("{}:{}", shortname, n->line());
        std::string       clickable;
        if (detail::is_terminal(os)) {
            clickable = make_clickable_file_line(n->file(), n->line(), file_display);
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
        double min_ms = ns_to_ms(n->exclusive_min());
        double max_ms = ns_to_ms(n->exclusive_max());
        double avg_ms = (n->call_count() > 0) ? (ns_to_ms(n->total_exclusive()) / n->call_count()) : 0.0;
        fmt::print(os, "{:6}   {:>10.3f}  (min {:>6.3f}  max {:>6.3f}  avg {:>6.3f})\n", "", excl_ms, min_ms, max_ms, avg_ms);
        if (!n->counters_total().empty()) {
            // print counters in one compact line
            fmt::print(os, "{:6}   Counters:", "");
            for (auto const &c : n->counters_total()) {
                uint64_t tot = c.second;
                uint64_t mn  = n->counters_min().at(c.first);
                uint64_t mx  = n->counters_max().at(c.first);
                double   avg = (n->call_count() > 0) ? static_cast<double>(tot) / n->call_count() : 0.0;
                fmt::print(os, " {}(tot={},min={},max={},avg={:.1f})", c.first, tot, mn, mx, avg);
            }
            fmt::print(os, "\n");
        }
    }

    std::list<std::shared_ptr<AggNode const>> children;
    for (auto const &c : n->children()) {
        children.push_back(c.second);
    }

    for (auto const &ch : children) {
        print_node_recursive(os, ch.get(), thread_total_ms, depth + 1, detailed);
    }
}

auto Profiler::thread_data() -> typename Profiler::ThreadMap & {
    return _global_data;
}
auto Profiler::thread_key() -> uint32_t {
#if defined _WIN32
    static_assert(sizeof(decltype(GetCurrentThreadId())) <= sizeof(uint32_t), "Thread handle too big to fit in protocol");
    return uint32_t(GetCurrentThreadId());
#elif defined __APPLE__
    uint64_t id;
    pthread_threadid_np(pthread_self(), &id);
    return uint32_t(id);
#elif defined __ANDROID__
    return (uint32_t)gettid();
#elif defined __linux__
    return static_cast<uint32_t>(syscall(SYS_gettid));
#elif defined __FreeBSD__
    long id;
    thr_self(&id);
    return id;
#elif defined __NetBSD__
    return _lwp_self();
#elif defined __DragonFly__
    return lwp_gettid();
#elif defined __OpenBSD__
    return getthrid();
#elif defined __QNX__
    return (uint32_t)gettid();
#elif defined __EMSCRIPTEN__
    // Not supported, but let it compile.
    return 0;
#else
    // To add support for a platform, retrieve and return the kernel thread identifier here.
    //
    // Note that pthread_t (as for example returned by pthread_self()) is *not* a kernel
    // thread identifier. It is a pointer to a library-allocated data structure instead.
    // Such pointers will be reused heavily, making the pthread_t non-unique. Additionally
    // a 64-bit pointer cannot be reliably truncated to 32 bits.
#    error "Unsupported platform!"
#endif
}

ScopedZone::ScopedZone(std::string const &name, std::string const &file, int line, std::string const &func) {
    Profiler::get_singleton().push(name, file, line, func);
}
ScopedZone::~ScopedZone() {
    std::fprintf(stderr, "Finalizing ScopedZone.");
    Profiler::get_singleton().pop();
}

} // namespace einsums::profile
