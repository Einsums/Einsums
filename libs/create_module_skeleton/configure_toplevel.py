#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import os
import sys

import cleanup_cmake

__truth = ["TRUE", "T", "YES", "ON", "Y", "1"]
__falsehood = ["FALSE", "F", "NO", "OFF", "N", "0"]


def configure_python(output_base, lib_name, **kwargs):
    if not kwargs["python"]:
        return
    base = os.path.dirname(__file__)

    export = ""

    config_python = {"CONFIGURE_PREAMBLE": "TRUE", "CONFIGURE_CLOSER": "TRUE"}

    if not os.path.exists(os.path.join(output_base, lib_name, ".is_python_lib")):
        with open(os.path.join(base, ".is_python_lib"), "r") as in_fp, open(
            os.path.join(output_base, lib_name, ".is_python_lib"), "w+"
        ) as out_fp:
            out_fp.write(in_fp.read())

    else:
        with open(os.path.join(output_base, lib_name, ".is_python_lib"), "r") as fp:
            for line in fp:
                toks = list(map(lambda x: x.strip(), line.split("=", 1)))
                if len(toks) < 2:
                    continue
                config_python[toks[0].upper()] = toks[1].upper()

    with open(os.path.join(base, "ExportAll.cpp.in"), "r") as fp:
        export = fp.read()

    with open(os.path.join(output_base, lib_name, "ExportAll.cpp.in"), "w+") as fp:
        fp.write(export.format(lib_name=lib_name, **kwargs))

    if (
        "CONFIGURE_PREAMBLE" not in config_python
        or config_python["CONFIGURE_PREAMBLE"].upper() not in __falsehood
    ):
        preamble = ""
        with open(os.path.join(base, "python_preamble.txt"), "r") as fp:
            preamble = fp.read()

        with open(os.path.join(output_base, lib_name, "preamble.txt"), "w+") as fp:
            fp.write(
                cleanup_cmake.cleanup_cmake(
                    preamble.format(lib_name=lib_name, **kwargs)
                )
            )

    if (
        "CONFIGURE_CLOSER" not in config_python
        or config_python["CONFIGURE_CLOSER"].upper() not in __falsehood
    ):
        preamble = ""
        with open(os.path.join(base, "python_closer.txt"), "r") as fp:
            preamble = fp.read()

        with open(os.path.join(output_base, lib_name, "closer.txt"), "w+") as fp:
            fp.write(
                cleanup_cmake.cleanup_cmake(
                    preamble.format(lib_name=lib_name, **kwargs)
                )
            )


def configure_cmake(output_base, lib_name, **kwargs):
    base = os.path.dirname(__file__)

    format = ""

    with open(os.path.join(base, "cmake_template.txt"), "r") as fp:
        format = fp.read()

    # Check for preamble.
    preamble = ""
    if os.path.exists(os.path.join(output_base, lib_name, "preamble.txt")):
        with open(os.path.join(output_base, lib_name, "preamble.txt"), "r") as fp:
            preamble = fp.read()

    closer = ""
    if os.path.exists(os.path.join(output_base, lib_name, "closer.txt")):
        with open(os.path.join(output_base, lib_name, "closer.txt"), "r") as fp:
            closer = fp.read()

    modules = "\n  ".join(
        sorted(
            filter(
                lambda x: os.path.exists(
                    os.path.join(output_base, lib_name, x, "CMakeLists.txt")
                ),
                os.listdir(os.path.join(output_base, lib_name)),
            )
        )
    )

    modules = modules.rstrip()

    with open(os.path.join(output_base, lib_name, "CMakeLists.txt"), "w+") as fp:
        fp.write(
            cleanup_cmake.cleanup_cmake(
                format.format(
                    modules=modules,
                    preamble=preamble,
                    closer=closer,
                    lib_name=lib_name,
                    **kwargs,
                )
            )
        )


def configure_module_docs(output_base, lib_name, **kwargs):
    base = os.path.dirname(__file__)

    format_str = ""

    with open(os.path.join(base, "modules.rst"), "r") as fp:
        format_str = fp.read()

    module_docs = "\n    ".join(
        sorted(
            map(
                lambda x: os.path.join("/libs", lib_name, x, "docs", "index.rst"),
                filter(
                    lambda x: os.path.exists(
                        os.path.join(output_base, lib_name, x, "CMakeLists.txt")
                    ),
                    os.listdir(os.path.join(output_base, lib_name)),
                ),
            )
        )
    )

    with open(os.path.join(output_base, lib_name, "modules.rst"), "w+") as fp:
        fp.write(
            format_str.format(
                module_docs=module_docs,
                lib_name=lib_name,
                section_head="".join("=" for i in f"{lib_name} modules"),
                **kwargs,
            )
        )
