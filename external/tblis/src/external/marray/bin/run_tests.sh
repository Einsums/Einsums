#!/bin/bash

status=$(bin/test 1>&2 && echo 0 || echo 1)

if [ -d force-cover-3.0 ]; then

    llvm-profdata merge default.profraw -o default.profdata
    llvm-cov show bin/test -instr-profile=default.profdata src/marray > coverage.txt
    python3 force-cover-3.0/fix_coverage.py coverage.txt

fi

exit $status
