//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Graph/BasicGraph.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/Utilities/

#include <string>

namespace einsums {

template <size_t Rank>
struct AbstractTensor {
  public:
    AbstractTensor(std::string const &name) : name_{name} {}

    std::string const &name() const { return name_; }

  private:
    std::string name_;
};

namespace graph {

using TensorNetwork = Graph<std::pair<std::string, std::vector<std::string>>, std::vector<std::string>>;

namespace detail {

// Base case.
template <typename AType>
void add_contraction_node(TensorNetwork *graph) {
    return;
}

template <typename AType, typename... AIndices, typename... IndicesTensors>
void add_contraction_node(TensorNetwork *graph, std::tuple<AIndices...> const &A_indices, AType const &A,
                          IndicesTensors &&...indices_tensors) {
    // Add a vertex for this tensor.
    auto new_vert = graph->emplace_vertex(A.name(), {std::string{AIndices::letter}...});

    constexpr auto A_unique = UniqueT<std::tuple<AIndices...>>();

    std::vector<std::string> A_unique_vec(std::tuple_size_v<decltype(A_unique)>);

    // Figure out the unique indices so we don't have a bunch of stray edges.
    for_sequence<std::tuple_size_v<decltype(A_unique)>>(
        [&](auto n) { A_unique_vec[(size_t)n] = std::tuple_element_t<(size_t)n, decltype(A_unique)>::letter; });

    // Add the edges connecting the tensor by indices.
    for (auto node : graph->vertices()) {
        if (node == new_vert) {
            continue;
        }
        for (auto from_ind : A_unique_vec) {
            for (auto to_ind : node->data().second) {
                if (from_ind == to_ind) {
                    graph->emplace_edge(new_vert, node, false, from_ind);
                    break;
                }
            }
        }
    }

    add_contraction_node(graph, std::forward<IndicesTensors>(indices_tensors)...);
}

} // namespace detail

template <typename... IndicesTensors>
TensorNetwork create_contraction(IndicesTensors &&...indices_tensors) {
    TensorNetwork out;

    detail::add_contraction_node(&out, std::forward<IndicesTensors>(indices_tensors)...);

    return out;
}

} // namespace graph

} // namespace einsums