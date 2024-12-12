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
        if item in ["Export.cpp", "Export.hpp"] and not kwargs["python"] :
            continue
        item_out = kwargs.get(item, item)

        if os.path.isdir(os.path.join(input_dir, item)):
            if not os.path.isdir(os.path.join(output_dir, item_out)):
                os.mkdir(os.path.join(output_dir, item_out))
            build_layer(
                os.path.join(input_dir, item), os.path.join(output_dir, item_out), **kwargs
            )
        elif not os.path.exists(os.path.join(output_dir, item_out)):
            format_str = ""
            try :
                with open(os.path.join(input_dir, item), "r") as fp:
                    format_str = fp.read()
                with open(os.path.join(output_dir, item_out), "w+") as fp:
                    fp.write(format_str.format(**kwargs))
            except KeyError as e:
                print(format_str)
                raise RuntimeError(f"File being parsed was {input_dir}/{item}.") from e


def build_structure(output_base, lib_name, module_name, python = False, **kwargs):
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
        docs_head = "".join("=" for i in lib_name + ' ' + module_name),
        readme_head = "".join("=" for i in module_name),
        export_header = f"{lib_name}/{module_name}/Export.hpp" if python else "",
        export_source = f"Export.cpp" if python else "",
        export_depends = "Einsums_Config" if python else "",
        python_footer = f"include(Einsums_ExtendWithPython)\neinsums_extend_with_python(${{EINSUMS_PYTHON_LIB_NAME}}_{module_name} ${{PYTHON_LIB_TYPE}})" if python else "",
        python = python,
        **kwargs
    )


