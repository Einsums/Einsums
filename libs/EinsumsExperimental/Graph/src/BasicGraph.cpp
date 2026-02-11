#include <Einsums/Errors/Error.hpp>
#include <Einsums/Graph/BasicGraph.hpp>

#include <stdexcept>

namespace einsums {
namespace graph {

#ifndef EINSUMS_WINDOWS
template struct Edge<void, void>;
template struct Edge<void, int>;
template struct Edge<void, double>;
template struct Vertex<void, void>;
template struct Vertex<void, int>;
template struct Vertex<void, double>;
template struct Graph<void, void>;
template struct Graph<void, int>;
template struct Graph<void, double>;
#endif

} // namespace graph
} // namespace einsums