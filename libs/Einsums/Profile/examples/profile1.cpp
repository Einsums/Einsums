//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime.hpp>

using namespace einsums::profile;

void microkernel() {
    LabeledSection("microkernel");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void pack() {
    LabeledSection("pack");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

void contract() {
    LabeledSection("contract");
    ScopedZone z("contract", __FILE__, __LINE__, __func__);
    pack();
    microkernel();
    microkernel();
}

int einsums_main() {
    {
        LabeledSection("main");
        std::thread t([] {
            LabeledSection("worker thread");
            contract();
        });

        contract();
        t.join();
    }

    // Print human-readable report
    Profiler::instance().print();

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}