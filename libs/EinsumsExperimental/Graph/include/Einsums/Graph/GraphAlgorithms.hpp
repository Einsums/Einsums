//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Graph/BasicGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <queue>
#include <random>
#include <stdexcept>

namespace einsums {
namespace graph {

template <typename Data, typename Weight>
Graph<Data, Weight> min_spanning_tree(Graph<Data, Weight> const &graph, bool random_start = false) {
    Graph<Data, Weight> out;

    std::priority_queue<SharedEdge<Data, Weight>, std::list<SharedEdge<Data, Weight>>, EdgeCompare<Data, Weight>> open_edges;

    SharedVertex<Data, Weight> curr_vert;

    std::uniform_int_distribution range(0, graph.num_vertices());

    while (out.num_vertices() != graph.num_vertices()) {
        if (random_start) {
            do {
                curr_vert = graph.vertex(range(einsums::random_engine));
            } while (out.has_vertex(curr_vert));
        } else {
            for (auto vertex : graph.vertices()) {
                if (out.has_vertex(vertex)) {
                    curr_vert = vertex;
                    break;
                }
            }

            EINSUMS_THROW_EXCEPTION(
                std::runtime_error,
                "All vertices have been added to the output graph, but the number of vertices in the input and output are different!");
        }

        // Loop the following while there are still edges to traverse.
        do {
            // Push the edges to the queue.
            for (auto edge : curr_vert.edges()) {
                open_edges.push(edge);
            }

            // Add the vertex to the graph.
            out.push_vertex(curr_vert->clone());

            // Find the smallest edge that connects to a new vertex.
            SharedEdge<Data, Weight> curr_edge;

            do {
                curr_edge = open_edges.top();

                open_edges.pop();
            } while (out.has_vertex(curr_edge->traverse(curr_vert)) && open_edges.size() > 0);

            // If we have all the nodes, then skip.
            if (out.has_vertex(curr_edge->traverse(curr_vert)) && open_edges.size() == 0) {
                break;
            }

            // Create a copy of the edge.
            auto new_edge    = curr_edge.clone();
            new_edge.start() = out.vertex_by_id(curr_edge->start()->lock()->id());
            new_edge.end()   = out.vertex_by_id(curr_edge->end()->lock()->id());

            out.push_edge(new_edge);

        } while (open_edges.size() > 0);
    }

    return out;
}

template <typename Data, typename Weight>
Tensor<double, 2> adjacency_matrix(Graph<Data, Weight> const &graph) {
    Tensor<double, 2> out{"Adjacency matrix", graph.num_vertices(), graph.num_vertices()};

    out.zero();

    for (int i = 0; i < graph.num_vertices(); i++) {
        auto vertex = graph.vertex(i);
        for (auto edge : vertex->edges()) {
            if (!edge.lock()->is_traversable(*vertex)) {
                continue;
            }
            auto goal = edge.lock()->traverse(*vertex).lock();

            for (int j = 0; j < graph.num_vertices(); j++) {
                if (*goal == *graph.vertex(j)) {
                    out(i, j) += 1.0;
                }
            }
        }
    }

    return out;
}

} // namespace graph
} // namespace einsums