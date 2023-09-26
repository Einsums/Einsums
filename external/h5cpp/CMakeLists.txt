# * v1.10.4-6-EAT6 on EAT branch is upstream v1.10.4-6 tag +2, so last upstream tag
#   plus extended array dimensions to support higher rank tensors plus deleter stuff.
# * find_package() is disabled since we need patched source
# * upstream CMakeLists.txt isn't useable and project is header-only, so to keep code
#   changes and build changes separate, we won't let FetchContent build (`SOURCE_SUBDIR
#   fake`) and will create the interface Einsums_h5cpp target after download.

FetchContent_Declare(
  h5cpp
  URL https://github.com/Einsums/h5cpp/archive/v1.10.4-6-EAT6.tar.gz
  URL_HASH SHA256=bc6370d2a2b11d9208c53d7815c2fbcfefc09f509f3a9f4d9cbd76a8c442feac
  SOURCE_SUBDIR fake
  OVERRIDE_FIND_PACKAGE
  )

