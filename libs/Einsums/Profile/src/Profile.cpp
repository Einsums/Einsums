//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/color.h>   // Enables text styles and colors
#include <fmt/ostream.h> // Enables fmt to write to std::ostream (e.g., ofstream)

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ostream>

#if defined(EINSUMS_WINDOWS)
#    include <io.h>
inline bool is_terminal(std::ostream const &os) {
    if (&os == &std::cout)
        return _isatty(_fileno(stdout));
    if (&os == &std::cerr)
        return _isatty(_fileno(stderr));
    return false;
}
#else
#    include <unistd.h>
inline bool is_terminal(std::ostream const &os) {
    if (&os == &std::cout)
        return isatty(fileno(stdout));
    if (&os == &std::cerr)
        return isatty(fileno(stderr));
    return false;
}
#endif

namespace einsums::profile::detail {

thread_local ThreadCallGraph thread_call_graph;

////////////////////////////////////////////////////////////////////
/// NodeMatrix
////////////////////////////////////////////////////////////////////
std::size_t NodeMatrix::rows() const noexcept {
    return _rows_size;
}

std::size_t NodeMatrix::cols() const noexcept {
    return _cols_size;
}

bool NodeMatrix::empty() const noexcept {
    return rows() == 0 || cols() == 0;
}

NodeId &NodeMatrix::prev_id(NodeId node_id) {
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _prev_ids[to_int(node_id)];
}

NodeId &NodeMatrix::next_id(CallSiteId callsite_id, NodeId node_id) {
    EINSUMS_ASSERT(to_int(callsite_id) < rows());
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _next_ids[to_int(callsite_id) + to_int(node_id) * _rows_capacity];
}

duration &NodeMatrix::time(NodeId node_id) {
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _times[to_int(node_id)];
}

std::unordered_map<std::string, uint64_t> &NodeMatrix::events(NodeId node_id) {
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _events[to_int(node_id)];
}

CallSiteInfo &NodeMatrix::callsite(CallSiteId callsite_id) {
    EINSUMS_ASSERT(to_int(callsite_id) < rows());
    return _callsites[to_int(callsite_id)];
}

NodeId const &NodeMatrix::prev_id(NodeId node_id) const {
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _prev_ids[to_int(node_id)];
}

NodeId const &NodeMatrix::next_id(CallSiteId callsite_id, NodeId node_id) const {
    EINSUMS_ASSERT(to_int(callsite_id) < rows());
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _next_ids[to_int(callsite_id) + to_int(node_id) * _rows_capacity];
}

duration const &NodeMatrix::time(NodeId node_id) const {
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _times[to_int(node_id)];
}

std::unordered_map<std::string, uint64_t> const &NodeMatrix::events(NodeId node_id) const {
    EINSUMS_ASSERT(to_int(node_id) < cols());
    return _events[to_int(node_id)];
}

CallSiteInfo const &NodeMatrix::callsite(CallSiteId callsite_id) const {
    EINSUMS_ASSERT(to_int(callsite_id) < rows());
    return _callsites[to_int(callsite_id)];
}

void NodeMatrix::resize(std::size_t new_rows, std::size_t new_cols) {
    bool const new_rows_over_capacity = new_rows > _rows_capacity;
    bool const new_cols_over_capacity = new_cols > _cols_capacity;
    bool const requires_reallocation  = new_rows_over_capacity || new_cols_over_capacity;

    // No reallocation case
    if (!requires_reallocation) {
        _rows_size = new_rows;
        _cols_size = new_cols;
        return;
    }

    // Reallocate
    std::size_t const new_rows_capacity = new_rows_over_capacity ? new_rows + NodeMatrix::row_growth_add : _rows_capacity;
    std::size_t const new_cols_capacity = new_cols_over_capacity ? new_cols * NodeMatrix::col_growth_mul : _cols_capacity;

    ArrayType<NodeId>                                    new_prev_ids(new_cols_capacity, NodeId::empty);
    ArrayType<NodeId>                                    new_next_ids(new_rows_capacity * new_cols_capacity, NodeId::empty);
    ArrayType<duration>                                  new_times(new_cols_capacity, duration{});
    ArrayType<std::unordered_map<std::string, uint64_t>> new_events(new_cols_capacity, std::unordered_map<std::string, uint64_t>{});
    ArrayType<CallSiteInfo>                              new_callsites(new_rows_capacity, CallSiteInfo{});

    // Copy old data
    for (std::size_t j = 0; j < _cols_size; ++j)
        new_prev_ids[j] = _prev_ids[j];
    for (std::size_t j = 0; j < _cols_size; ++j)
        for (std::size_t i = 0; i < _rows_size; ++i)
            new_next_ids[i + j * new_rows_capacity] = _next_ids[i + j * _rows_capacity];
    for (std::size_t j = 0; j < _cols_size; ++j)
        new_times[j] = _times[j];
    for (std::size_t j = 0; j < _cols_size; ++j)
        new_events[j] = _events[j];
    for (std::size_t i = 0; i < _rows_size; ++i)
        new_callsites[i] = _callsites[i];

    // Assign new data
    _prev_ids  = std::move(new_prev_ids);
    _next_ids  = std::move(new_next_ids);
    _times     = std::move(new_times);
    _events    = std::move(new_events);
    _callsites = std::move(new_callsites);

    _rows_size     = new_rows;
    _cols_size     = new_cols;
    _rows_capacity = new_rows_capacity;
    _cols_capacity = new_cols_capacity;
}

////////////////////////////////////////////////////////////////////
/// Profiler
////////////////////////////////////////////////////////////////////

Profiler::Profiler() : _main_thread_id(std::this_thread::get_id()), _thread_counter(0), _performance_counter(PerformanceCounter::create()) {
}

Profiler::~Profiler() = default;

Profiler &Profiler::get() {
    static Profiler instance;
    return instance;
}

void Profiler::upload_this_thread() {
    thread_call_graph.upload_results(false);
}

void Profiler::format_results(std::ostream &out, Style const &style) {
    // Walk through all the threads and make sure they have updated the Profiler
    for (auto thread : _thread_call_graphs)
        thread->upload_results(false);

    upload_this_thread();

    return format_available_results(out, style);
}

void Profiler::add_thread_call_graph(ThreadCallGraph *thread_call_graph) {
    _thread_call_graphs.push_back(thread_call_graph);
}

void Profiler::call_graph_add(std::thread::id thread_id) {
    std::lock_guard const lock(_call_graph_mutex);

    auto const [it, emplaced] = _call_graph_info.try_emplace(thread_id);

    if (emplaced) {
        it->second.readable_id = (thread_id == _main_thread_id) ? 0 : ++_thread_counter;
    }

    it->second.lifetimes.emplace_back();
}

void Profiler::call_graph_upload(std::thread::id thread_id, NodeMatrix &&info, bool joined) {
    std::lock_guard const lock(_call_graph_mutex);

    auto &lifetime  = _call_graph_info.at(thread_id).lifetimes.back();
    lifetime.mat    = std::move(info);
    lifetime.joined = joined;
}

bool Profiler::results_are_empty() {
    for (auto const &[thread_id, thread_lifetimes] : _call_graph_info) {
        for (auto const &lifetime : thread_lifetimes.lifetimes) {
            if (lifetime.joined == false || !lifetime.mat.empty())
                return false;
        }
    }
    return true;
}

namespace {
std::string format_call_site(std::string const &file, std::size_t line, std::string const &function) {
    std::filesystem::path path(file);
    std::string           filename = path.filename().string();
    std::string           res;
    res.reserve(filename.size() + function.size() + 10);
    fmt::format_to(std::back_inserter(res), "{}:{}, {}()", filename, line, function);
    return res;
}

// Internal helper
template <typename... Args>
void safe_print(std::ostream &os, fmt::text_style style, std::string_view format_str, Args &&...args) {
    std::string output;
    if (is_terminal(os)) {
        output = fmt::format(style, fmt::runtime(format_str), std::forward<Args>(args)...);
    } else {
        output = fmt::format(fmt::runtime(format_str), std::forward<Args>(args)...);
    }
    fmt::print(os, "{}", output);
}

constexpr fmt::emphasis no_style = static_cast<fmt::emphasis>(0);

} // namespace

void Profiler::format_available_results(std::ostream &out, Style const &style) {
    std::lock_guard const lock(_call_graph_mutex);

    std::vector<FormattedRow> rows;

    // Format header
    safe_print(out, emphasis::bold | fg(color::cyan), "\n{:-^110}\n", " EINSUMS PROFILING RESULTS ");

    // Format using strftime-style format string
    // Get system time
    auto now = std::chrono::system_clock::now();
    // Convert to time_t
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // Convert to local time (non-thread-safe)
    std::tm *local_tm = std::localtime(&now_c);
    // Print using fmt (safe even with non-thread-safe tm*)
    safe_print(out, no_style, "\nReporting time: {:%Y-%m-%d %H:%M:%S} local time\n", *local_tm);

    for (auto const &[thread_id, thread_lifetimes] : _call_graph_info) {
        for (std::size_t reuse = 0; reuse < thread_lifetimes.lifetimes.size(); ++reuse) {
            auto const       &mat         = thread_lifetimes.lifetimes[reuse].mat;
            bool const        joined      = thread_lifetimes.lifetimes[reuse].joined;
            std::size_t const readable_id = thread_lifetimes.readable_id;

            rows.clear();
            rows.reserve(mat.cols());

            std::string const thread_str      = (readable_id == 0) ? "main" : std::to_string(readable_id);
            bool const        thread_uploaded = !mat.empty();

            // Format thread header
            safe_print(out, emphasis::bold | fg(color::cyan), "\n# Thread [ {} ] (reuse {})", thread_str, std::to_string(reuse));

            // Format thread status
            safe_print(out, joined ? emphasis::bold | fg(color::green) : emphasis::bold | fg(color::magenta), " ({})",
                       joined ? "joined" : "running");

            // Early escape for lifetimes that haven't uploaded yet
            if (!thread_uploaded) {
                safe_print(out, no_style, "\n");
                continue;
            }

            // Format thread runtime
            ms const runtime = mat.time(NodeId::root);
            safe_print(out, emphasis::bold | fg(color::light_blue), " (runtime -> {:.2f} ms)\n", runtime.count());

            // Gather call graph data in a digestible format
            mat.root_apply_recursively([&](CallSiteId callsite_id, NodeId node_id, std::size_t depth) {
                if (callsite_id == CallSiteId::empty)
                    return;

                auto const  &callsite   = mat.callsite(callsite_id);
                auto const  &time       = mat.time(node_id);
                auto const  &events     = mat.events(node_id);
                double const percentage = time / runtime;

                rows.push_back(FormattedRow{callsite, time, events, depth, percentage});
            });

            // Format call graph columns row by row
            // The array is of length 4 for 1. percentage string, 2. time string, 3. label string, 4. call site information
            // This could be extended in the future to performance counter information.
            std::vector<std::array<std::string, 4>> rows_str;
            rows_str.reserve(rows.size());

            for (auto const &row : rows) {
                auto percentage_str = std::string(style.indent * row.depth, ' ');
                fmt::format_to(std::back_inserter(percentage_str), " - {:.2f}%", row.percentage * 100);

                auto time_str     = fmt::format("{:.2f} ms", ms(row.time).count());
                auto label_str    = row.callsite.label;
                auto callsite_str = format_call_site(row.callsite.file, row.callsite.line, row.callsite.function);

                rows_str.push_back({std::move(percentage_str), std::move(time_str), std::move(label_str), std::move(callsite_str)});
            }

            // Gather column widths for alignment
            std::size_t width_percentage = 0, width_time = 0, width_label = 0, width_callsite = 0;
            for (auto const &row : rows_str) {
                width_percentage = std::max(width_percentage, row[0].size());
                width_time       = std::max(width_time, row[1].size());
                width_label      = std::max(width_label, row[2].size());
                width_callsite   = std::max(width_callsite, row[3].size());
            }

            assert(rows.size() == rows_str.size());

            // Print out column labels
            safe_print(out, no_style, "{:^{}} | {:^{}} | {:^{}} | {:^{}} |\n", "Percent", width_percentage, "Time", width_time, "Label",
                       width_label, "Location", width_callsite);
            safe_print(out, no_style, "{} | {} | {} | {} |\n", std::string(width_percentage, '-'), std::string(width_time, '-'),
                       std::string(width_label, '-'), std::string(width_callsite, '-'));

            // Format result with colors and alignment
            for (std::size_t i = 0; i < rows.size(); ++i) {
                bool const color_row_red    = rows[i].percentage > style.cutoff_red;
                bool const color_row_yellow = rows[i].percentage > style.cutoff_yellow;
                bool const color_row_gray   = rows[i].percentage < style.cutoff_gray;

                fmt::text_style text_color;
                if (color_row_red)
                    text_color = fg(color::red);
                else if (color_row_yellow)
                    text_color = fg(color::yellow);
                else if (color_row_gray)
                    text_color = fg(color::gray);

                safe_print(out, text_color, "{:-<{}} | {:>{}} | {:>{}} | {:<{}} |\n", rows_str[i][0], width_percentage, rows_str[i][1],
                           width_time, rows_str[i][2], width_label, rows_str[i][3], width_callsite);
            }
        }
    }
}

PerformanceCounter &Profiler::get_performance_counter() const {
    return *_performance_counter.get();
}

////////////////////////////////////////////////////////////////////
/// Thread Call Graph
////////////////////////////////////////////////////////////////////

ThreadCallGraph::ThreadCallGraph() {
    Profiler::get().call_graph_add(_thread_id);
    Profiler::get().add_thread_call_graph(this);
    create_root_node();
}

ThreadCallGraph::~ThreadCallGraph() {
    // upload_results(true);
}

NodeId ThreadCallGraph::create_root_node() {
    NodeId const prev_node_id = _current_node_id;
    _current_node_id          = NodeId::root;

    _mat.grow_nodes();
    _mat.prev_id(_current_node_id) = prev_node_id;

    return _current_node_id;
}

NodeId ThreadCallGraph::create_node(CallSiteId callsite_id) {
    NodeId const prev_node_id = _current_node_id;
    _current_node_id          = NodeId(_mat.cols());

    _mat.grow_nodes();
    _mat.prev_id(_current_node_id)          = prev_node_id;
    _mat.next_id(callsite_id, prev_node_id) = _current_node_id;

    return _current_node_id;
}

void ThreadCallGraph::upload_results(bool joined) {
    _mat.time(NodeId::root) = clock::now() - _entry_time_point;

    Profiler::get().call_graph_upload(_thread_id, NodeMatrix(_mat), joined);
}

NodeId ThreadCallGraph::traverse_forward(CallSiteId callsite_id) {
    NodeId const next_node_id = _mat.next_id(callsite_id, _current_node_id);
    if (next_node_id == NodeId::empty)
        return create_node(callsite_id);
    return _current_node_id = next_node_id;
}

void ThreadCallGraph::traverse_back() {
    _current_node_id = _mat.prev_id(_current_node_id);
}

void ThreadCallGraph::record(duration time, std::unordered_map<std::string, uint64_t> &events) {
    _mat.time(_current_node_id) += time;
    // ensure the vector is of the right length
    auto &thread_events = _mat.events(_current_node_id);
    // if "key" doesn't exist in thread_events it is automatically added and
    // assigned a value of 0.
    for (auto const &[key, value] : events) {
        thread_events[key] += value;
    }
}

CallSiteId ThreadCallGraph::callsite_add(CallSiteInfo const &info) {
    CallSiteId const new_call_site_id = CallSiteId(_mat.rows());

    _mat.grow_callsites();
    _mat.callsite(new_call_site_id) = info;

    return new_call_site_id;
}

////////////////////////////////////////////////////////////////////
/// Call Site Marker
////////////////////////////////////////////////////////////////////

CallSite::CallSite(CallSiteInfo const &info) {
    _id = thread_call_graph.callsite_add(info);
}

////////////////////////////////////////////////////////////////////
/// Timer
////////////////////////////////////////////////////////////////////
Timer::Timer(CallSiteId id) : _performance_counter(Profiler::get().get_performance_counter()) {
    thread_call_graph.traverse_forward(id);
    _counters.start.resize(_performance_counter.nevents());
    _counters.end.resize(_performance_counter.nevents());

    _entry = clock::now();
    _performance_counter.capture(_counters.start);
}

void Timer::finish() {
    _performance_counter.capture(_counters.end);
    auto duration = clock::now() - _entry;
    _performance_counter.delta(_counters.start, _counters.end);
    auto mapped = _performance_counter.to_event_map(_counters.end);

    thread_call_graph.record(duration, mapped);
    thread_call_graph.traverse_back();
}

} // namespace einsums::profile::detail
