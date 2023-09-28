# * Not using hptt is just asking for slow sorts.
# * find_package() is disabled since we need patched source
# * hptt is built as an object library and not installed

FetchContent_Declare(
  hptt
  URL https://github.com/Einsums/hptt/archive/v1.0.5+24.tar.gz
  URL_HASH SHA256=6cfe70bbd026005e059d0a4031c21eb6ac7c2d14f6e1914cc2b8c9ce2a8b616d
  SYSTEM
  OVERRIDE_FIND_PACKAGE
  )

FetchContent_MakeAvailable(hptt)
