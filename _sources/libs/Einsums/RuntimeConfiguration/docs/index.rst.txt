..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_RuntimeConfiguration:

====================
RuntimeConfiguration
====================

This module contains facilities for runtime configuration. It is considered to be an internal
module, though one function in particular may be of use to users.

See the :ref:`API reference <modules_Einsums_RuntimeConfiguration_api>` of this module for more
details.

Public Symbols
--------------

.. cpp:function:: void register_arguments(std::function<void(argparse::ArgumentParser &)> func)

    Adds a function that registers command line arguments to the list of start-up functions. This
    should be called before Einsums is initialized. During initialization, the function given to this
    will be called, allowing it to set up command line arguments. These arguments will then be processed.
    This is an example of what can be done with this, taken from the BufferAllocator module.

    .. code:: C++

        void add_Einsums_BufferAllocator_arguments(argparse::ArgumentParser &parser) {
            // Get the config maps for making the options available to the program.
            auto &global_config = GlobalConfigMap::get_singleton();
            auto &global_string = global_config.get_string_map()->get_value();

            // Add the argument.
            parser.add_argument("--einsums:buffer-size")
                .default_value("4MB")
                .help("Total size of buffers allocated for tensor contractions.")
                .store_into(global_string["buffer-size"]);

            // Attach an observer to look for changes to this argument.
            global_config.attach(detail::Einsums_BufferAllocator_vars::update_max_size);
        }

    :param func: The function to run during initialization.