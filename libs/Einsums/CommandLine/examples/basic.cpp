//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

// -------------------------- Example (guarded) ---------------------------- //

#include <Einsums/CommandLine/CommandLine.hpp>

using namespace einsums::cl;

OptionCategory Cat{"General"};
Flag           Verbose{"verbose", {'v'}, "Enable verbose output", Cat};
Opt<int>       Iters{"iters",
                     {'i'},
               5,
               "Iteration count",
               Cat,
               Visibility::Normal,
               Occurrence::Optional,
               ValueExpected::ValueRequired,
               RangeBetween(1, 1000000)};

Subcommand       Tools{"tools", "Tooling subcommands"};
OptionCategory   ToolsCat{"Tools"};
Opt<std::string> ToolName{"tool", {'t'}, "Which tool", ToolsCat};

List<std::string> Inputs{"inputs", Positional{}, "Input files"};

int main(int argc, char **argv) {
    auto pr = parse_with_config(argc, argv, "ein-tool", "1.2.3", "config.json");
    if (!pr.ok)
        return pr.exit_code;
    fmt::print("verbose={} iters={} inputs={}\n", Verbose.get(), Iters.get(), Inputs.values().size());
    return 0;
}
