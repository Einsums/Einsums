//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Profile/Timer.hpp>

#include <cassert>
#include <chrono>
#include <map>
#include <mutex>
#include <omp.h>
#include <vector>

namespace einsums::profile {

namespace detail {

std::mutex lock;

struct TimerDetail {
    // Description of the timing block
    std::string name{"(no name)"};

    // Accumulated runtime
    clock::duration total_time{0};

    // Number of times the timer has been called
    size_t total_calls{0};

    std::shared_ptr<TimerDetail>                        parent{nullptr};
    std::map<std::string, std::shared_ptr<TimerDetail>> children{};
    std::vector<std::string>                            order{};

    time_point start_time;
};

std::shared_ptr<TimerDetail> current_timer{nullptr};
std::shared_ptr<TimerDetail> root{nullptr};

} // namespace detail

namespace detail {
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void print_timer_info(std::shared_ptr<TimerDetail> timer, std::FILE *fp) { // NOLINT
    if (timer != root) {
        std::string buffer;
        if (timer->total_calls != 0) {
            buffer = fmt::format("{:>5} : {:>5} calls : {:>5} per call", duration_cast<milliseconds>(timer->total_time), timer->total_calls,
                                 duration_cast<milliseconds>(timer->total_time) / timer->total_calls);
        } else {
            buffer = "total_calls == 0!!!";
        }
        int width = 70 - print::current_indent_level();
        if (width < 0) {
            width = 0;
        }
        fprintln(fp, "{0:<{1}} : {3: <{4}}{2}", buffer, width, timer->name, "", print::current_indent_level());
    } else {
        fprintln(fp);
        fprintln(fp);
        fprintln(fp, "Timing information:");
        fprintln(fp);
    }

    if (!timer->children.empty()) {
        print::indent();

        for (auto &child : timer->order) {
            print_timer_info(timer->children[child], fp);
        }

        print::deindent();
    }
}

} // namespace detail

void initialize() {
    using namespace detail;
    root              = std::make_shared<TimerDetail>();
    root->name        = "Total Run Time";
    root->total_calls = 1;

    current_timer = root;

    // Determine timer overhead
    for (size_t i = 0; i < 1000; i++) {
        push("Timer Overhead");
        pop();
    }
}

void finalize() {
    using namespace detail;
    assert(root == current_timer);
    root.reset();
    current_timer.reset();
}

void report(std::string const &fname, bool append) {
    std::FILE *fp = std::fopen(fname.c_str(), append ? "w+" : "w");

    detail::print_timer_info(detail::root, fp);

    std::fflush(fp);
    std::fclose(fp);
}

void push(std::string name) {
    using namespace detail;
    // assert(current_timer != nullptr);
    static bool already_warned{false};

    std::lock_guard guard(lock);

    if (omp_get_thread_num() == 0) {
        if (omp_in_parallel()) {
            name = fmt::format("{} (master thread only)", name);
        }

        if (!current_timer) {
            if (already_warned == false) {
                println("Timer::push: Timer was not initialized prior to calling `push`. This is the only warning you will receive.");
                already_warned = true;
            }
            return;
        }

        if (current_timer->children.contains(name) == false) {
            current_timer->children[name]         = std::make_shared<TimerDetail>();
            current_timer->children[name]->name   = name;
            current_timer->children[name]->parent = current_timer;
            current_timer->order.push_back(name);
        }

        current_timer             = current_timer->children[name];
        current_timer->start_time = clock::now();
    }
}

void pop() {
    using namespace detail;
    static bool already_warned{false};

    std::lock_guard guard(lock);

    if (omp_get_thread_num() == 0) {
        if (current_timer == nullptr) {
            if (already_warned == false) {
                println(
                    "Timer::pop: current_timer is already nullptr; something might be wrong. This is the only warning you will receive.");
                already_warned = true;
            }
            return;
        }

        current_timer->total_time += clock::now() - current_timer->start_time;
        current_timer->total_calls++;
        current_timer = current_timer->parent;
    }
}

void pop(duration elapsed) {
    using namespace detail;
    static bool already_warned{false};

    std::lock_guard<std::mutex> guard(lock);

    if (omp_get_thread_num() == 0) {
        if (current_timer == nullptr) {
            if (already_warned == false) {
                println(
                    "Timer::pop: current_timer is already nullptr; something might be wrong. This is the only warning you will receive.");
                already_warned = true;
            }
            return;
        }

        current_timer->total_time += elapsed;
        current_timer->total_calls++;
        current_timer = current_timer->parent;
    }
}

} // namespace einsums::profile