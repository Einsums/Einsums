#include <Einsums/Graph/BasicGraph.hpp>
#include <Einsums/Graph/GraphAlgorithms.hpp>
#include <Einsums/Tensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Adjacency matrix", "[graphs]") {
    using namespace einsums;
    using namespace einsums::graph;

    Graph<void, void> graph;

    // Making the seven bridges of Koenigberg.
    graph.emplace_vertex();
    graph.emplace_vertex();
    graph.emplace_vertex();
    graph.emplace_vertex();

    // Two bridges from the first landmass to the second.
    graph.emplace_edge(graph.vertex(0), graph.vertex(1), false);
    graph.emplace_edge(graph.vertex(0), graph.vertex(1), false);

    // One bridge from the first to the third.
    graph.emplace_edge(graph.vertex(0), graph.vertex(2), false);

    // Two bridges from the first to the fourth.
    graph.emplace_edge(graph.vertex(0), graph.vertex(3), false);
    graph.emplace_edge(graph.vertex(0), graph.vertex(3), false);

    // One from the second to the third.
    graph.emplace_edge(graph.vertex(1), graph.vertex(2), false);

    // One from the third to the fourth.
    graph.emplace_edge(graph.vertex(2), graph.vertex(3), false);

    Tensor<double, 2> expected{"Expected result", 4, 4};

    expected.zero();

    expected(0, 1) = 2.0;
    expected(1, 0) = 2.0;
    expected(0, 2) = 1.0;
    expected(2, 0) = 1.0;
    expected(0, 3) = 2.0;
    expected(3, 0) = 2.0;
    expected(1, 2) = 1.0;
    expected(2, 1) = 1.0;
    expected(2, 3) = 1.0;
    expected(3, 2) = 1.0;

    auto result = adjacency_matrix(graph);

    println(result);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            REQUIRE_THAT(result(i, j), Catch::Matchers::WithinAbs(expected(i, j), 1e-6));
        }
    }
}

TEST_CASE("Creation and deletion", "[graphs]") {

    using namespace einsums;
    using namespace einsums::graph;

    Graph<void, void> *graph = new Graph<void, void>();

    // Making the seven bridges of Koenigberg.
    graph->emplace_vertex();
    graph->emplace_vertex();
    graph->emplace_vertex();
    graph->emplace_vertex();

    // Two bridges from the first landmass to the second.
    graph->emplace_edge(graph->vertex(0), graph->vertex(1), false);
    graph->emplace_edge(graph->vertex(0), graph->vertex(1), false);

    // One bridge from the first to the third.
    graph->emplace_edge(graph->vertex(0), graph->vertex(2), false);

    // Two bridges from the first to the fourth.
    graph->emplace_edge(graph->vertex(0), graph->vertex(3), false);
    graph->emplace_edge(graph->vertex(0), graph->vertex(3), false);

    // One from the second to the third.
    graph->emplace_edge(graph->vertex(1), graph->vertex(2), false);

    // One from the third to the fourth.
    graph->emplace_edge(graph->vertex(2), graph->vertex(3), false);

    REQUIRE_NOTHROW(delete graph);
}