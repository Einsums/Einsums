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
        item_out = item
        if item in kwargs :
            item_out = kwargs[item]
        if os.path.isdir(os.path.join(input_dir, item)):
            if not os.path.isdir(os.path.join(output_dir, item_out)):
                os.mkdir(os.path.join(output_dir, item_out))
            build_layer(
                os.path.join(input_dir, item), os.path.join(output_dir, item_out), **kwargs
            )
        elif not os.path.exists(os.path.join(output_dir, item_out)):
            format_str = ""
            with open(os.path.join(input_dir, item), "r") as fp:
                format_str = fp.read()
            with open(os.path.join(output_dir, item_out), "w+") as fp:
                fp.write(format_str.format(**kwargs))


def build_structure(output_base, lib_name, module_name, **kwargs):
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
        docs_head = "=".join(lib_name + ' ' + module_name),
        readme_head = "=".join(module_name),
        **kwargs
    )


