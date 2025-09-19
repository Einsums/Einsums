#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.

from pathlib import Path
import re

FUNC_RE = re.compile(r'^\s*(function|macro)\s*\(\s*([A-Za-z0-9_]+)', re.IGNORECASE)


def extract_blocks(text):
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("#:"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("#:"):
                block.append(lines[i].split(":#", 1)[-1].lstrip() if lines[i].startswith("#:#") else lines[i].lstrip()[
                    2:].lstrip())
                i += 1
            # lookahead for the next function/macro name
            name = None
            j = i
            while j < len(lines) and name is None:
                m = FUNC_RE.match(lines[j])
                if m:
                    name = m.group(2)
                j += 1
            out.append((name, "\n".join(block)))
        else:
            i += 1
    return out


def generate(src_dir, dst_dir):
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for cm in Path(src_dir).glob("*.cmake"):
        blocks = extract_blocks(cm.read_text(encoding="utf-8"))
        if not blocks:
            continue
        parts = [".. _cmake_{0}:\n".format(cm.stem),
                 cm.stem, "=" * len(cm.stem), ""]
        for name, rst in blocks:
            parts.append(rst.rstrip() + "\n")
        (dst / f"{cm.stem}.rst").write_text("\n".join(parts), encoding="utf-8")
