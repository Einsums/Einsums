# * Not using hptt is just asking for slow sorts.
# * find_package() is disabled since we need patched source

FetchContent_Declare(
  hptt
  GIT_REPOSITORY https://github.com/Einsums/hptt.git
  GIT_TAG v1.0.5+23
  SYSTEM
  OVERRIDE_FIND_PACKAGE
  )

FetchContent_MakeAvailable(hptt)
