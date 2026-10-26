//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Compat/FmtAlternatives.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/TypeSupport/InsertionOrderedMap.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stack>
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
#    ifdef EINSUMS_HAVE_TRACY
#        include "TracyWinFamily.hpp"
#    endif
#else
#    include <cstring>
#    include <pthread.h>
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
struct EINSUMS_EXPORT AggNode {
  public:
    AggNode() = default;
    explicit AggNode(std::string const &n);

    AggNode(AggNode const &copy)     = default;
    AggNode(AggNode &&move) noexcept = default;

    AggNode &operator=(AggNode const &copy)     = default;
    AggNode &operator=(AggNode &&move) noexcept = default;

    ~AggNode();

    [[nodiscard]] std::string const &name() const noexcept;
    [[nodiscard]] std::string       &name() noexcept;

    [[nodiscard]] std::string const &file() const noexcept;
    [[nodiscard]] std::string       &file() noexcept;

    [[nodiscard]] int line() const noexcept;

    [[nodiscard]] std::string const &function() const noexcept;
    [[nodiscard]] std::string       &function() noexcept;

    [[nodiscard]] uint64_t  call_count() const noexcept;
    [[nodiscard]] ns const &total_exclusive() const noexcept;

    [[nodiscard]] int64_t total_exclusive_mean() const noexcept;

    [[nodiscard]] int64_t total_exclusive_M2() const noexcept;

    [[nodiscard]] ns const &exclusive_min() const noexcept;

    [[nodiscard]] ns const &exclusive_max() const noexcept;

    [[nodiscard]] std::map<std::string, uint64_t> const &counters_total() const noexcept;
    [[nodiscard]] std::map<std::string, uint64_t> const &counters_min() const noexcept;
    [[nodiscard]] std::map<std::string, uint64_t> const &counters_max() const noexcept;

    [[nodiscard]] std::shared_ptr<AggNode> find_or_add(std::string const &name);

    [[nodiscard]] InsertionOrderedMap<std::string, std::shared_ptr<AggNode>> const &children() const noexcept;

    void set_name(std::string const &name);

    void set_file(std::string const &fname);

    void set_line(int line) noexcept;

    void set_function(std::string const &func);

    void inc_call_count() noexcept;
    void set_call_count(uint64_t count) noexcept;

    void inc_exclusive(ns const &time);

  private:
    std::string name_;
    std::string file_;
    int         line_ = 0;
    std::string function_;

    // counts and times (ns)
    uint64_t call_count_ = 0;
    ns       total_exclusive_{0};

    // These are needed to compute a running standard deviation. values are in nanoseconds.
    int64_t total_exclusive_mean_{0};
    int64_t total_exclusive_M2_{0};

    // min/max for exclusive time
    ns exclusive_min_{std::numeric_limits<int64_t>::max()};
    ns exclusive_max_{0};

    // counters aggregate: name -> total/min/max
    std::map<std::string, uint64_t> counters_total_;
    std::map<std::string, uint64_t> counters_min_;
    std::map<std::string, uint64_t> counters_max_;

    InsertionOrderedMap<std::string, std::shared_ptr<AggNode>> children_;
};

// ---------------------- Active frame ----------------------
struct EINSUMS_EXPORT ActiveFrame {
  public:
    ActiveFrame() = delete;
    ActiveFrame(std::string const &name, TimePoint const &start, std::string const &file, int line, std::string const &function);

    ActiveFrame(ActiveFrame const &copy)     = default;
    ActiveFrame(ActiveFrame &&move) noexcept = default;

    ActiveFrame &operator=(ActiveFrame const &copy)     = default;
    ActiveFrame &operator=(ActiveFrame &&move) noexcept = default;

    ~ActiveFrame();

    [[nodiscard]] std::string const &name() const noexcept;

    [[nodiscard]] TimePoint const &start() const noexcept;

    [[nodiscard]] ns const &child_time() const noexcept;

    [[nodiscard]] std::string const &file() const noexcept;

    [[nodiscard]] int line() const noexcept;

    [[nodiscard]] std::string const &function() const noexcept;

    [[nodiscard]] std::string &name() noexcept;

    [[nodiscard]] TimePoint &start() noexcept;

    [[nodiscard]] ns &child_time() noexcept;

    [[nodiscard]] std::string &file() noexcept;

    [[nodiscard]] int &line() noexcept;

    [[nodiscard]] std::string &function() noexcept;

    void inc_child_time(ns const &time);

  private:
    std::string name_;
    TimePoint   start_;
    ns          child_time_{0};
    // optional source location
    std::string file_; // full path
    int         line_ = 0;
    std::string function_;
};

struct EINSUMS_EXPORT ActiveStack final {
  private:
    class PrivateConstructorStuff {};

  public:
    using iterator       = typename std::vector<ActiveFrame>::iterator;
    using const_iterator = typename std::vector<ActiveFrame>::const_iterator;

    constexpr explicit ActiveStack(PrivateConstructorStuff const &) : ActiveStack() {}

    static ActiveStack &get_multiton();

    ActiveStack(ActiveStack const &) = delete;
    ActiveStack(ActiveStack &&)      = delete;

    ActiveStack &operator=(ActiveStack const &) = delete;
    ActiveStack &operator=(ActiveStack &&)      = delete;
    
    ~ActiveStack();

    [[nodiscard]] ActiveFrame const &top() const;
    [[nodiscard]] ActiveFrame       &top();

    [[nodiscard]] bool empty() const;

    [[nodiscard]] size_t size() const;

    void push(ActiveFrame const &value);
    // No move push since it's highly likely that the strings come from a different library (breaks Windows).

    // Not nodiscard, since the value can safely be ignored (and often is).
    template <typename... Args>
    ActiveFrame &emplace(Args &&...args) {
        return stack_.emplace_back(std::forward<Args>(args)...);
    }

    void pop();

    [[nodiscard]] iterator       begin();
    [[nodiscard]] const_iterator begin() const;
    [[nodiscard]] const_iterator cbegin() const noexcept;

    [[nodiscard]] iterator       end() noexcept;
    [[nodiscard]] const_iterator end() const noexcept;
    [[nodiscard]] const_iterator cend() const noexcept;

    // And no swap.

  private:
    ActiveStack() = default;

    std::vector<ActiveFrame> stack_;
};

#    ifdef EINSUMS_HAVE_TRACY
struct EINSUMS_EXPORT TracyStack final {
  private:
    class PrivateConstructorStuff {};

  public:
    constexpr explicit TracyStack(PrivateConstructorStuff const &) : TracyStack() {}

    static TracyStack &get_multiton();

    TracyStack(TracyStack const &) = delete;
    TracyStack(TracyStack &&)      = delete;

    TracyStack &operator=(TracyStack const &) = delete;
    TracyStack &operator=(TracyStack &&)      = delete;

    [[nodiscard]] tracy::ScopedZone const &top() const;
    [[nodiscard]] tracy::ScopedZone       &top();

    [[nodiscard]] bool empty() const;

    [[nodiscard]] size_t size() const;

    void push(tracy::ScopedZone const &value);
    // No move push since it's highly likely that the strings come from a different library (breaks Windows).

    // Not nodiscard, since the value can safely be ignored (and often is).
    template <typename... Args>
    std::unique_ptr<tracy::ScopedZone> &emplace(Args &&...args) {
        return stack_.emplace(std::make_unique<tracy::ScopedZone>(std::forward<Args>(args)...));
    }

    void pop();

    // And no swap.

  private:
    TracyStack() = default;

    std::stack<tracy::ScopedZone, std::vector<std::unique_ptr<tracy::ScopedZone>>> stack_;
};
#    endif

// ---------------------- Profiler class ----------------------
struct EINSUMS_EXPORT Profiler final {
    EINSUMS_SINGLETON_DEF(Profiler)
  public:
    ~Profiler();
  
    // Start a timer region. Optionally provide file/line/func (if available).
    void push(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "");

    // Stop timer region
    void pop();

    // Print default compact report (exclusive time, percent, name, file:line clickable, func)
    // detailed -> show min/max/avg and counters
    void print(bool detailed = false, std::ostream &os = std::cout);

    // JSON & CSV exporters (optional)
    bool export_json(std::string const &path = "einsums_profile.json");

  private:
    Profiler() = default;

    void write_node_json(std::ostream &ofs, AggNode const &n, int indent);

    // recursive pretty printer (compact)
    void print_node_recursive(std::ostream &os, AggNode const *n, double thread_total_ms, int depth, bool detailed);

    // ------------------ thread/local data ------------------

    // ------------------ global aggregated storage keyed by thread id ------------------
    using ThreadMap = std::unordered_map<uint32_t, AggNode>;
    auto        thread_data() -> ThreadMap &;
    static auto thread_key() -> uint32_t;

    // thread-sum helper
    ThreadMap  _global_data;
    std::mutex _mutex;
};

// ---------------------- Scoped helper ----------------------
struct EINSUMS_EXPORT ScopedZone {
    explicit ScopedZone(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "");
    ~ScopedZone();
};

#    define LabeledSection(name_format, ...)                                                                                               \
        ::einsums::profile::ScopedZone const EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(einsums::format(name_format, ##__VA_ARGS__),          \
                                                                                     __FILE__, __LINE__, __func__)
#    define LabeledSection0() LabeledSection(__func__)
#    if defined(EINSUMS_WITH_PROFILER_INTERNAL)
#        define LabeledSectionInternal(name_format, ...)                                                                                   \
            ::einsums::profile::ScopedZone const EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(einsums::format(name_format, ##__VA_ARGS__),      \
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
