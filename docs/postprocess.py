#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
#  Copyright (c) The Einsums Developers. All rights reserved.
#  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""
Post-processes HTML and Latex files output by Sphinx.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', help='file mode', choices=('html', 'tex'))
    parser.add_argument('file', nargs='+', help='input file(s)')
    args = parser.parse_args()

    mode = args.mode

    for fn in args.file:
        with open(fn, encoding="utf-8") as f:
            if mode == 'html':
                lines = process_html(fn, f.readlines())
            elif mode == 'tex':
                lines = process_tex(f.readlines())

        with open(fn, 'w', encoding="utf-8") as f:
            f.write("".join(lines))


def process_html(fn, lines):
    return lines


def process_tex(lines):
    """
    Remove unnecessary section titles from the LaTeX file.

    """
    new_lines = []
    for line in lines:
        if (line.startswith(r'\section{einsums.')
                or line.startswith(r'\subsection{einsums.')
                or line.startswith(r'\subsubsection{einsums.')
                or line.startswith(r'\paragraph{einsums.')
                or line.startswith(r'\subparagraph{einsums.')
        ):
            pass  # skip!
        else:
            new_lines.append(line)
    return new_lines


if __name__ == "__main__":
    main()
