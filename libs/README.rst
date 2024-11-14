..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

This directory holds the modularized libraries that Einsums is built upon. Those libraries
can be seen as independent modules with clear dependencies and no cycles.

The tool ``create_module_skeleton.py`` can be used to generate a basic module
skeleton. The structure of this skeleton should be as follows:

* ``<lib_name>/``

  * ``README.rst``
  * ``CMakeLists.txt``
  * ``cmake``
  * ``docs/``

    * ``index.rst``

  * ``examples/``

    * ``CMakeLists.txt``

  * ``include/``

    * ``Einsums/``

      * ``<lib_name>``

  * ``src/``

    * ``CMakeLists.txt``

  * ``tests/``

    * ``CMakeLists.txt``
    * ``unit/``

      * ``CMakeLists.txt``

    * ``regressions/``

      * ``CMakeLists.txt``

    * ``performance/``

      * ``CMakeLists.txt``

A ``README.rst`` should always be included. It should explain the basic purpose of
the library and a link to the generated documentation.

The ``include`` directory should only contain the headers that other libraries need.
Private headers should be placed under the ``src`` directory. This allows for
clear separation of the public and private API. The ``cmake`` subdirectory may include additional |cmake|_
scripts needed to generate the respective build configurations.

Documentation is placed in the ``docs`` folder. An empty skeleton for the index
is created by the creation script, which is picked up by the main build system and will be part of the
generated documentation.
