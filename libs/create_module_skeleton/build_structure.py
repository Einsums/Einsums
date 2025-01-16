#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import os
import sys


def parse_template(filename, output_file, **kwargs):
    format_str = ""

    with open(filename, "r") as fp:
        format_str = fp.read()

    with open(output_file, "w+") as fp:
        fp.write(format_str.format(**kwargs))


def build_layer(input_dir, output_dir, **kwargs):
    for item in os.listdir(input_dir):
        # Skip exports if we don't need them.
        if item in ["Export.cpp"] and not kwargs["python"]:
            continue
        item_out = kwargs.get(item, item)

        if os.path.splitext(item_out)[1] == ".fstring" :
            item_out = os.path.splitext(item_out)[0]

        if os.path.isdir(os.path.join(input_dir, item)):
            if not os.path.isdir(os.path.join(output_dir, item_out)):
                os.mkdir(os.path.join(output_dir, item_out))
            build_layer(
                os.path.join(input_dir, item), os.path.join(output_dir, item_out), **kwargs
            )
        elif not os.path.exists(os.path.join(output_dir, item_out)):
            format_str = ""
            try:
                with open(os.path.join(input_dir, item), "r") as fp:
                    format_str = fp.read()
                with open(os.path.join(output_dir, item_out), "w+") as fp:
                    fp.write(format_str.format(**kwargs))
            except KeyError as e:
                print(format_str)
                raise RuntimeError(f"File being parsed was {input_dir}/{item}.") from e
            except ValueError as e:
                print(format_str)
                print(e)
                raise RuntimeError(f"File being parsed was {input_dir}/{item}.") from e


def build_structure(output_base, lib_name, module_name, python=False, **kwargs):
    base = os.path.dirname(__file__)

    if not os.path.exists(os.path.join(output_base, lib_name)):
        os.mkdir(os.path.join(output_base, lib_name))

    if not os.path.exists(os.path.join(output_base, lib_name, module_name)):
        os.mkdir(os.path.join(output_base, lib_name, module_name))

    build_layer(
        os.path.join(base, "template_top"),
        os.path.join(output_base, lib_name, module_name),
        module_name=module_name,
        lib_name=lib_name,
        docs_head="".join("=" for i in lib_name + ' ' + module_name),
        readme_head="".join("=" for i in module_name),
        export_source="Export.cpp" if python else "",
        export_depends="" if lib_name == "Einsums" else "Einsums",
        python_footer=f"include(Einsums_ExtendWithPython)\neinsums_extend_with_python_headers(${{EINSUMS_PYTHON_LIB_NAME}}_{module_name})" if python else "",
        python=python,
        python_deps="pybind11::embed" if python else "",
        gpu_head="if(EINSUMS_WITH_GPU_SUPPORT)" if kwargs["gpu"] else "",
        gpu_foot="endif()" if kwargs["gpu"] else "",
        **kwargs
    )
