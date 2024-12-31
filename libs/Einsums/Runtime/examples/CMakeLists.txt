#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if (EINSUMS_WITH_EXAMPLES)
    einsums_add_pseudo_target(Examples.Modules.Einsums.Runtime)
    einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.Einsums.Runtime)
    if (EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
        einsums_add_pseudo_target(Tests.Examples.Modules.Einsums.Runtime)
        einsums_add_pseudo_dependencies(Tests.Examples.Modules Tests.Examples.Modules.Einsums.Runtime)
    endif ()
else ()
    return()
endif ()

set(example_programs HelloWorld HelloWorld2)

foreach (example_program ${example_programs})
    set(sources ${example_program}.cpp)

    source_group("Source Files" FILES ${sources})

    # add example executable
    einsums_add_executable(
            ${example_program} INTERNAL_FLAGS
            SOURCES ${sources} ${${example_program}_FLAGS}
            FOLDER "Examples/Modules/Runtime"
    )

    einsums_add_example_target_dependencies("Modules.Einsums.Runtime" ${example_program})

    if (PIKA_WITH_TESTS AND PIKA_WITH_TESTS_EXAMPLES)
        einsums_add_example_test(
                "Modules.Einsums.Runtime" ${example_program} ${${example_program}_PARAMETERS}
        )
    endif ()
endforeach ()
