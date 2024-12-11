import os
import sys


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

    modules = "\n  ".join(
        filter(
            lambda x: os.path.isdir(os.path.join(output_base, lib_name, x)),
            os.listdir(os.path.join(output_base, lib_name)),
        )
    )

    modules = modules.rstrip()

    with open(os.path.join(output_base, lib_name, "CMakeLists.txt"), "w+") as fp:
        if kwargs["gpu"] :
            print("if(EINSUMS_WITH_GPU_SUPPORT)", file = fp)

        fp.write(
            format.format(
                modules=modules, preamble=preamble, lib_name=lib_name, **kwargs
            )
        )

        if kwargs["gpu"] :
            print("endif()")


def configure_module_docs(output_base, lib_name, **kwargs):
    base = os.path.dirname(__file__)

    format_str = ""

    with open(os.path.join(base, "modules.rst"), "r") as fp:
        format_str = fp.read()

    module_docs = "\n    ".join(
        map(
            lambda x: os.path.join("/libs", lib_name, x, "docs", "index.rst"),
            filter(
                lambda x: os.path.isdir(os.path.join(output_base, lib_name, x)),
                os.listdir(os.path.join(output_base, lib_name)),
            ),
        )
    )

    with open(os.path.join(output_base, lib_name, "modules.rst"), "w+") as fp:
        fp.write(
            format_str.format(
                module_docs=module_docs,
                lib_name=lib_name,
                section_head="=".join(f"{lib_name} modules"),
                **kwargs,
            )
        )
