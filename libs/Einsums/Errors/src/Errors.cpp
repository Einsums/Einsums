#include <Einsums/Errors/Error.hpp>

#define EXCEPTION_DEF(name, base) \
einsums::name::name(const char *what) : base(what) {} \
einsums::name::name(const std::string &what) : base(what) {}

EXCEPTION_DEF(dimension_error, std::invalid_argument)
EXCEPTION_DEF(tensor_compat_error, std::logic_error)
EXCEPTION_DEF(num_argument_error, std::invalid_argument)
EXCEPTION_DEF(not_enough_args, einsums::num_argument_error)
EXCEPTION_DEF(too_many_args, einsums::num_argument_error)
EXCEPTION_DEF(access_denied, std::logic_error)
EXCEPTION_DEF(todo_error, std::logic_error)
EXCEPTION_DEF(bad_logic, std::logic_error)
