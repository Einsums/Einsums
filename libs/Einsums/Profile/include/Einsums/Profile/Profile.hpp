//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>
#include <Einsums/TypeSupport/InsertionOrderedMap.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef EINSUMS_HAVE_TRACY
#    include <tracy/Tracy.hpp>
#endif

#if defined _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <malloc.h>
#    include <windows.h>

#    include "TracyWinFamily.hpp"
#else
#    include <pthread.h>
#    include <string.h>
#    include <unistd.h>
#endif

#ifdef __linux__
#    ifdef __ANDROID__
#        include <sys/types.h>
#    else
#        include <sys/syscall.h>
#    endif
#    include <fcntl.h>
#elif defined __FreeBSD__
#    include <sys/thr.h>
#elif defined __NetBSD__
#    include <lwp.h>
#elif defined __DragonFly__
#    include <sys/lwp.h>
#elif defined __QNX__
#    include <process.h>
#    include <sys/neutrino.h>
#endif

namespace einsums::profile {

#if defined(EINSUMS_HAVE_PROFILER)

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using ns        = std::chrono::nanoseconds; // NOLINT

// ---------------------- Agg node ----------------------
struct AggNode {
    std::string name;
    std::string file;
    int         line = 0;
    std::string function;

    // counts and times (ns)
    uint64_t call_count = 0;
    ns       total_exclusive{0};

    // These are needed to compute a running standard deviation. values are in nanoseconds.
    int64_t total_exclusive_mean{0};
    int64_t total_exclusive_M2{0};

    // min/max for exclusive time
    ns exclusive_min{std::numeric_limits<int64_t>::max()};
    ns exclusive_max{0};

    // counters aggregate: name -> total/min/max
    std::map<std::string, uint64_t> counters_total;
    std::map<std::string, uint64_t> counters_min;
    std::map<std::string, uint64_t> counters_max;

    InsertionOrderedMap<std::string, std::unique_ptr<AggNode>> children;

    AggNode() = default;
    explicit AggNode(std::string n) : name(std::move(n)) {}
};

// ---------------------- Active frame ----------------------
struct ActiveFrame {
    std::string name;
    TimePoint   start;
    ns          child_time{0};
    // optional source location
    std::string file; // full path
    int         line = 0;
    std::string function;
};

// ---------------------- Profiler class ----------------------
struct EINSUMS_EXPORT Profiler {
    static auto instance() -> Profiler & {
        static Profiler p;
        return p;
    }

    // Start a timer region. Optionally provide file/line/func (if available).
    void push(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        auto now = Clock::now();

#    ifdef EINSUMS_HAVE_TRACY
        // dynamic runtime zone name using ScopedZone
        auto z =
            std::make_unique<tracy::ScopedZone>(line, file.c_str(), file.size(), func.c_str(), func.size(), name.c_str(), name.size(), 1);
        thread_tracy_zones().push_back(std::move(z));
#    endif

        active_stack().push_back(
            ActiveFrame{.name = name, .start = now, .child_time = ns{0}, .file = file, .line = line, .function = func});
    }

    // Stop timer region
    void pop() {
        auto now = Clock::now();
        if (active_stack().empty())
            return;
        ActiveFrame const frame = active_stack().back();
        active_stack().pop_back();

        ns const duration  = std::chrono::duration_cast<ns>(now - frame.start);
        ns const exclusive = duration - frame.child_time;

        // build path from root to this node
        std::vector<std::string> path;
        for (auto &f : active_stack())
            path.push_back(f.name);
        path.push_back(frame.name);

        // collect counter deltas (if any)
        std::map<std::string, uint64_t> const deltas;

#    ifdef EINSUMS_HAVE_TRACY
        // pop the tracy zone
        if (!thread_tracy_zones().empty())
            thread_tracy_zones().pop_back();
#    endif

        // update aggregated data
        {
            std::lock_guard const lock(_mutex);
            auto                 &root = thread_data()[thread_key()];
            AggNode              *cur  = &root;
            for (auto &p : path) {
                auto it = cur->children.find(p);
                if (it == cur->children.end()) {
                    cur->children[p] = std::make_unique<AggNode>(p);
                    it               = cur->children.find(p);
                }
                cur = it->second.get();
            }

            cur->file     = frame.file;
            cur->line     = frame.line;
            cur->function = frame.function;

            cur->call_count += 1;
            cur->total_exclusive += exclusive;

            int64_t const delta = exclusive.count() - cur->total_exclusive_mean;
            println("function {} delta {} exclusive {} total mean {}", cur->function, delta, exclusive, cur->total_exclusive_mean);
            cur->total_exclusive_mean += delta / int64_t(cur->call_count);
            int64_t const delta2 = exclusive.count() - cur->total_exclusive_mean;
            cur->total_exclusive_M2 += delta * delta2;

            if (exclusive < cur->exclusive_min)
                cur->exclusive_min = exclusive;
            if (exclusive > cur->exclusive_max)
                cur->exclusive_max = exclusive;

            // merge counters
            for (auto &kv : deltas) {
                std::string const &ename = kv.first;
                uint64_t const     val   = kv.second;
                cur->counters_total[ename] += val;
                auto itmin = cur->counters_min.find(ename);
                if (itmin == cur->counters_min.end()) {
                    cur->counters_min[ename] = val;
                    cur->counters_max[ename] = val;
                } else {
                    if (val < cur->counters_min[ename])
                        cur->counters_min[ename] = val;
                    if (val > cur->counters_max[ename])
                        cur->counters_max[ename] = val;
                }
            }
        }

        // add duration to parent's child_time so parent exclusive deducts it
        if (!active_stack().empty())
            active_stack().back().child_time += duration;
    }

    // Print default compact report (exclusive time, percent, name, file:line clickable, func)
    // detailed -> show min/max/avg and counters
    void print(bool detailed = false, std::ostream &os = std::cout);

    // JSON & CSV exporters (optional)
    auto export_json(std::string const &path = "einsums_profile.json") -> std::optional<std::string>;

  private:
    Profiler() = default;

    void write_node_json(std::ostream &ofs, AggNode const &n, int indent);

    // recursive pretty printer (compact)
    void print_node_recursive(std::ostream &os, AggNode const *n, double thread_total_ms, int depth, bool detailed);

    // wrapper that supplies first-level call
    void print_node_recursive(std::ostream &os, AggNode *n, double thread_total_ms, int depth, bool detailed) {
        print_node_recursive(os, static_cast<AggNode const *>(n), thread_total_ms, depth, detailed);
    }

    // ------------------ thread/local data ------------------
    static auto active_stack() -> std::vector<ActiveFrame> & {
        thread_local std::vector<ActiveFrame> s;
        return s;
    }

#    ifdef EINSUMS_HAVE_TRACY
    static auto thread_tracy_zones() -> std::vector<std::unique_ptr<tracy::ScopedZone>> & {
        thread_local std::vector<std::unique_ptr<tracy::ScopedZone>> v;
        return v;
    }
#    endif

    // ------------------ global aggregated storage keyed by thread id ------------------
    using ThreadMap = std::unordered_map<uint32_t, AggNode>;
    auto        thread_data() -> ThreadMap        &{ return _global_data; }
    static auto thread_key() -> uint32_t {
#    if defined _WIN32
        static_assert(sizeof(decltype(GetCurrentThreadId())) <= sizeof(uint32_t), "Thread handle too big to fit in protocol");
        return uint32_t(GetCurrentThreadId());
#    elif defined __APPLE__
        uint64_t id;
        pthread_threadid_np(pthread_self(), &id);
        return uint32_t(id);
#    elif defined __ANDROID__
        return (uint32_t)gettid();
#    elif defined __linux__
        return static_cast<uint32_t>(syscall(SYS_gettid));
#    elif defined __FreeBSD__
        long id;
        thr_self(&id);
        return id;
#    elif defined __NetBSD__
        return _lwp_self();
#    elif defined __DragonFly__
        return lwp_gettid();
#    elif defined __OpenBSD__
        return getthrid();
#    elif defined __QNX__
        return (uint32_t)gettid();
#    elif defined __EMSCRIPTEN__
        // Not supported, but let it compile.
        return 0;
#    else
        // To add support for a platform, retrieve and return the kernel thread identifier here.
        //
        // Note that pthread_t (as for example returned by pthread_self()) is *not* a kernel
        // thread identifier. It is a pointer to a library-allocated data structure instead.
        // Such pointers will be reused heavily, making the pthread_t non-unique. Additionally
        // a 64-bit pointer cannot be reliably truncated to 32 bits.
#        error "Unsupported platform!"
#    endif
    }

    // thread-sum helper
    ThreadMap  _global_data;
    std::mutex _mutex;
};

// ---------------------- Scoped helper ----------------------
struct ScopedZone {
    explicit ScopedZone(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        Profiler::instance().push(name, file, line, func);
    }
    ~ScopedZone() { Profiler::instance().pop(); }
};

#    define LabeledSection(name_format, ...)                                                                                               \
        ::einsums::profile::ScopedZone const EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(fmt::format(name_format, ##__VA_ARGS__), __FILE__,    \
                                                                                     __LINE__, __func__)
#    define LabeledSection0() LabeledSection(__func__)
#    if defined(EINSUMS_WITH_PROFILER_INTERNAL)
#        define LabeledSectionInternal(name_format, ...)                                                                                   \
            ::einsums::profile::ScopedZone const EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(fmt::format(name_format, ##__VA_ARGS__),          \
                                                                                         __FILE__, __LINE__, __func__)
#        define LabeledSectionInternal0() LabeledSectionInternal(__func__)
#    else
#        define LabeledSectionInternal(...)
#        define LabeledSectionInternal0()
#    endif
#else
#    define LabeledSection(...)
#    define LabeledSection0()
#    define LabeledSectionInternal(...)
#    define LabeledSectionInternal0()
#endif

} // namespace einsums::profile
