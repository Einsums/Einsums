# Future Ideas for Einsums

## Overall Ideas

Here are ideas for Einsums that either haven't been assigned to a submodule or
do not necessarily fit within a single submodule.

## Per-Submodule Ideas

Ideas that fit to a particular submodule are listed here. This includes ideas for
new submodules.

### Logging

* When work begins on a distributed Einsums it might be usefuly to include a TCP or UDP sink.
  This would require a server to accept the logging data to be provided, too.
* ~~Need to provide `EINSUMS_LOG_TRACE` and `EINSUMS_LOG_DEBUG` macros. While it is good to have
  detailed logging information available I fear it can add unwarranted overhead. If these macros
  are provided that can be easily enabled and disabled at compile-time. Enable them in
  test and development builds, but disable them (at least `EINSUMS_LOG_TRACE`) for production.~~

### Profile

### RuntimeConfiguration

* Need to provide access to argc and argv with the `--einsums:*` command-line options
  filtered out. Then the testing harness needs to be updated to pass argc
  and argv from the RuntimeConfiguration. Currently, it is being given the original
  argc and argv from `main` and Catch2 is reporting `Unrecognised token: --einsum`
  error. The test is still permitted to run but having a clean test output would be better.

### TensorAlgebra

* Need a way to test which algorithm is being used by `einsum`.

## Testing Harness

* Once the work to process access to argc and argv from the RuntimeConfiguration the
  testing harness needs to be updated to use them.
