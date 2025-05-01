//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#include <Einsums/JSON.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>

#include <chrono>
#include <iostream>
#include <thread>

using namespace einsums;
using namespace einsums::json;

void simulate_task(std::shared_ptr<JSONWriter::Entry> entry, std::string const &name, int sleep_ms) {
    entry->write("task", name);
    entry->write("start_time", static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    entry->write("end_time", static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()));
    entry->write("duration_ms", sleep_ms);
}

int einsums_main() {
    JSONWriter json;

    // Top-level record
    auto session = json.record("session");
    session->write("type", "profiling");
    session->write("timestamp", "2025-05-01T12:00:00Z");

    // Add a list of tasks
    auto task_array = session->array("tasks");

    // Simulate tasks
    for (int i = 0; i < 3; ++i) {
        auto task = task_array->object();
        simulate_task(task, "task_" + std::to_string(i), 100 + i * 20);
    }

    // Nested object
    auto metadata = session->object("metadata");
    metadata->write("host", "localhost");
    metadata->write("version", "1.0.0");

    // Output pretty-printed JSON
    std::cout << json.str(true) << std::endl;

    finalize();

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
