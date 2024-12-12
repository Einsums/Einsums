import os
import sys


def configure_python(output_base, lib_name, **kwargs):
    if not kwargs["python"]:
        return
    base = os.path.dirname(__file__)

    export = ""

    with open(os.path.join(base, "ExportAll.cpp.in"), "r") as fp:
        export = fp.read()

    with open(os.path.join(output_base, lib_name, "ExportAll.cpp.in"), "w+") as fp:
        fp.write(export.format(lib_name = lib_name, **kwargs))

    if not os.path.exists(os.path.join(output_base, lib_name, "preamble.txt")):
        preamble = ""
        with open(os.path.join(base, "python_preamble.txt"), "r") as fp:
            preamble = fp.read()

        with open(os.path.join(output_base, lib_name, "preamble.txt"), "w+") as fp:
            fp.write(preamble.format(lib_name = lib_name, **kwargs))

    if not os.path.exists(os.path.join(output_base, lib_name, "closer.txt")):
        preamble = ""
        with open(os.path.join(base, "python_closer.txt"), "r") as fp:
            preamble = fp.read()

        with open(os.path.join(output_base, lib_name, "closer.txt"), "w+") as fp:
            fp.write(preamble.format(lib_name = lib_name, **kwargs))


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
        filter(
            lambda x: os.path.isdir(os.path.join(output_base, lib_name, x)),
            os.listdir(os.path.join(output_base, lib_name)),
        )
    )

    modules = modules.rstrip()

    with open(os.path.join(output_base, lib_name, "CMakeLists.txt"), "w+") as fp:
        if kwargs["gpu"]:
            print("if(EINSUMS_WITH_GPU_SUPPORT)", file=fp)

        fp.write(
            format.format(
                modules=modules,
                preamble=preamble,
                closer=closer,
                lib_name=lib_name,
                **kwargs,
            )
        )

        if kwargs["gpu"]:
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
                section_head="".join("=" for i in f"{lib_name} modules"),
                **kwargs,
            )
        )
