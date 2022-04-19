#pragma once

#include <h5cpp/core>

// Items in this namespace contribute to the global state of the program
namespace einsums::State {

extern h5::fd_t data;
extern h5::fd_t checkpoint_file;

} // namespace einsums::State