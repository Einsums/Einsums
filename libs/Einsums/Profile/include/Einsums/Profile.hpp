//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Enum.hpp>
#include <Einsums/Profile/Detail/CPUFrequency.hpp>
#include <Einsums/Profile/Detail/PerformanceCounter.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <type_traits>

namespace einsums::profile::detail {

using clock      = std::chrono::steady_clock;
using duration   = clock::duration;
using time_point = clock::time_point;

using ms = std::chrono::duration<double, std::chrono::milliseconds::period>;

// Type-safe IDs
template <ScopedEnum E>
[[nodiscard]] constexpr auto to_int(E value) noexcept {
    return static_cast<std::underlying_type_t<E>>(value);
}

using IdType = std::uint32_t;

enum class CallSiteId : IdType { empty = to_int(CallSiteId(-1)) };
enum class NodeId : IdType { root = 0, empty = to_int(NodeId(-1)) };

struct CallSiteInfo {
    char const *file;
    char const *function;
    std::string label;
    int         line;
};

// Formatting
struct Style {
    int indent = 2;

    double cutoff_red    = 0.40; // >40% of runtime
    double cutoff_yellow = 0.20; // >20% of runtime
    double cutoff_gray   = 0.01; // <1% of runtime
};

struct FormattedRow {
    CallSiteInfo                              callsite;
    duration                                  time;
    std::unordered_map<std::string, uint64_t> events;
    std::size_t                               depth;
    double                                    percentage;
};

// Call graph core structure
struct EINSUMS_EXPORT NodeMatrix {
  private:
    template <typename T>
    using ArrayType = std::vector<T>;

    constexpr static std::size_t col_growth_mul = 2;
    constexpr static std::size_t row_growth_add = 4;

    ArrayType<NodeId> _prev_ids;
    ArrayType<NodeId> _next_ids;

    ArrayType<duration>                                  _times;
    ArrayType<std::unordered_map<std::string, uint64_t>> _events;

    ArrayType<CallSiteInfo> _callsites;

    std::size_t _rows_size;
    std::size_t _cols_size;
    std::size_t _rows_capacity;
    std::size_t _cols_capacity;

  public:
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;

    bool empty() const noexcept;

    NodeId                                    &prev_id(NodeId node_id);
    NodeId                                    &next_id(CallSiteId callsite_id, NodeId node_id);
    duration                                  &time(NodeId node_id);
    std::unordered_map<std::string, uint64_t> &events(NodeId node_id);
    CallSiteInfo                              &callsite(CallSiteId callsite_id);

    NodeId const                                    &prev_id(NodeId node_id) const;
    NodeId const                                    &next_id(CallSiteId callsite_id, NodeId node_id) const;
    duration const                                  &time(NodeId node_id) const;
    std::unordered_map<std::string, uint64_t> const &events(NodeId node_id) const;
    CallSiteInfo const                              &callsite(CallSiteId callsite_id) const;

    void resize(std::size_t new_rows, std::size_t new_cols);

    void grow_callsites() { resize(_rows_size + 1, _cols_size); }
    void grow_nodes() { resize(_rows_size, _cols_size + 1); }

    template <class Func, std::enable_if_t<std::is_invocable_v<Func, CallSiteId, NodeId, std::size_t>, bool> = true>
    void node_apply_recursively(CallSiteId callsite_id, NodeId node_id, Func func, std::size_t depth) const {
        func(callsite_id, node_id, depth);

        for (std::size_t i = 0; i < rows(); ++i) {
            CallSiteId const next_callsite_id = CallSiteId(i);
            NodeId const     next_node_id     = next_id(next_callsite_id, node_id);
            if (next_node_id != NodeId::empty)
                node_apply_recursively(next_callsite_id, next_node_id, func, depth + 1);
        }
        // 'node_is' corresponds to a matrix column, to iterate over all
        // "next" nodes we iterate rows (callsites) in a column
    }

    template <class Func, std::enable_if_t<std::is_invocable_v<Func, CallSiteId, NodeId, std::size_t>, bool> = true>
    void root_apply_recursively(Func func) const {
        if (!_rows_size || !_cols_size)
            return; // possibly redundant

        func(CallSiteId::empty, NodeId::root, 0);

        for (std::size_t i = 0; i < rows(); ++i) {
            CallSiteId const next_callsite_id = CallSiteId(i);
            NodeId const     next_node_id     = next_id(next_callsite_id, NodeId::root);
            if (next_node_id != NodeId::empty)
                node_apply_recursively(next_callsite_id, next_node_id, func, 1);
        }
    }
};

// Profiler

struct ThreadLifetimeData {
    NodeMatrix mat;
    bool       joined = false;
};

struct ThreadIdData {
    std::vector<ThreadLifetimeData> lifetimes;
    std::size_t                     readable_id;
};

struct ThreadCallGraph;

struct EINSUMS_EXPORT Profiler {
  private:
    using CallGraphStorage = std::unordered_map<std::thread::id, ThreadIdData>;

    friend struct ThreadCallGraph;
    CallGraphStorage _call_graph_info;
    std::mutex       _call_graph_mutex;

    std::thread::id _main_thread_id;
    std::size_t     _thread_counter;

    std::mutex _setter_mutex;

    std::vector<ThreadCallGraph *> _thread_call_graphs;

    std::unique_ptr<PerformanceCounter> _performance_counter;

    bool results_are_empty();

    void format_available_results(std::ostream &out, Style const &style = Style{});

    void call_graph_add(std::thread::id thread_id);

    void call_graph_upload(std::thread::id thread_id, NodeMatrix &&info, bool joined);

  public:
    Profiler();
    ~Profiler();

    void upload_this_thread();

    void format_results(std::ostream &out, Style const &style = Style{});

    void add_thread_call_graph(ThreadCallGraph *thread_call_graph);

    static Profiler &get();
};

// Is this even needed?
// extern Profiler profiler;

struct EINSUMS_EXPORT ThreadCallGraph {
    NodeMatrix      _mat;
    NodeId          _current_node_id  = NodeId::empty;
    time_point      _entry_time_point = clock::now();
    std::thread::id _thread_id        = std::this_thread::get_id();

    NodeId create_root_node();
    NodeId create_node(CallSiteId callsite_id);
    void   upload_results(bool joined);

    ThreadCallGraph();
    ~ThreadCallGraph();

    NodeId traverse_forward(CallSiteId callsite_id);
    void   traverse_back();

    void record_time(duration time);

    CallSiteId callsite_add(CallSiteInfo const &info);
};

struct EINSUMS_EXPORT CallSite {
    CallSite(CallSiteInfo const &info);
    CallSiteId get_id() const noexcept { return _id; }

  private:
    CallSiteId _id;
};

struct EINSUMS_EXPORT Timer {
    Timer(CallSiteId id);

    void finish() const;

  private:
    time_point entry = clock::now();
};

struct ScopeTimer : Timer {
    ScopeTimer(CallSiteId id) : Timer(id) {}

    constexpr operator bool() const noexcept { return true; }

    ~ScopeTimer() { finish(); }
};

} // namespace einsums::profile::detail

#define einsums_profile_uuid(varname_) EINSUMS_PP_CAT(varname_, __LINE__)

#define EINSUMS_PROFILE_SCOPE(label_, ...)                                                                                                 \
    constexpr bool einsums_profile_uuid(einsums_profile_macro_guard_) = true;                                                              \
    static_assert(einsums_profile_uuid(einsums_profile_macro_guard_), "EINSUMS_PROFILE is a multi-line macro.");                           \
                                                                                                                                           \
    const thread_local einsums::profile::detail::CallSite einsums_profile_uuid(einsums_profile_callsite_)(                                 \
        einsums::profile::detail::CallSiteInfo(__FILE__, __func__, fmt::format(label_, ##__VA_ARGS__), __LINE__));                         \
                                                                                                                                           \
    const einsums::profile::detail::ScopeTimer einsums_profile_uuid(einsums_profile_scope_timer_) {                                        \
        einsums_profile_uuid(einsums_profile_callsite_).get_id()                                                                           \
    }

#define EINSUMS_PROFILE(label_)                                                                                                            \
    constexpr bool einsums_profile_uuid(einsums_profile_macro_guard_) = true;                                                              \
    static_assert(einsums_profile_uuid(einsums_profile_macro_guard_), "EINSUMS_PROFILER is a multi-line macro.");                          \
                                                                                                                                           \
    const thread_local einsums::profile::detail::CallSite einsums_profile_uuid(einsums_profile_callsite_)(                                 \
        einsums::profile::detail::CallSiteInfo{__FILE__, __func__, label_, __LINE__});                                                     \
                                                                                                                                           \
    if constexpr (const einsums::profile::detail::ScopeTimer einsums_profile_uuid(einsums_profile_scope_timer_){                           \
                      einsums_profile_uuid(einsums_profile_callsite_).get_id()})

// 'if constexpr (timer)' allows this macro to "capture" the scope of the following expression

#define EINSUMS_PROFILE_BEGIN(segment_, label_, ...)                                                                                       \
    const thread_local einsums::profile::detail::CallSite einsums_profile_callsite_##segment_(                                             \
        einsums::profile::detail::CallSiteInfo{__FILE__, __func__, fmt::format(label_, ##__VA_ARGS__), __LINE__});                         \
                                                                                                                                           \
    const einsums::profile::detail::Timer einsums_profile_timer_##segment_ {                                                               \
        einsums_profile_callsite_##segment_.get_id()                                                                                       \
    }

#define EINSUMS_PROFILE_END(segment_) einsums_profile_timer_##segment_.finish()
