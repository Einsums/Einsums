//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace einsums {
namespace graph {

template <typename Data, typename Weight>
struct Vertex;
template <typename Data, typename Weight>
struct Edge;
template <typename Data, typename Weight>
struct Graph;

template <typename Data, typename Weight>
using SharedVertex = std::shared_ptr<Vertex<Data, Weight>>;
template <typename Data, typename Weight>
using WeakVertex = std::weak_ptr<Vertex<Data, Weight>>;
template <typename Data, typename Weight>
using SharedEdge = std::shared_ptr<Edge<Data, Weight>>;
template <typename Data, typename Weight>
using WeakEdge = std::weak_ptr<Edge<Data, Weight>>;

template <typename Data, typename Weight>
struct Vertex {
  public:
    Vertex() : serial_id_{0}, edges_(), data_() {}

    template <typename... Args>
    Vertex(size_t serial_id, Args &&...args) : serial_id_{serial_id}, edges_(), data_(std::forward<Args>(args)...) {}

    Vertex(Vertex const &other) : serial_id_{other.id()}, edges_(), data_(other.data()) {}

    virtual ~Vertex() { edges_.clear(); }

    SharedVertex<Data, Weight> clone() const { return std::make_shared<Vertex<Data, Weight>>(id(), data()); }

    bool operator==(Vertex const &other) const { return id() == other.id(); }

    size_t id() const { return serial_id_; }

    void set_id(size_t new_id) { serial_id_ = new_id; }

    std::vector<WeakEdge<Data, Weight>> &edges() { return edges_; }

    std::vector<WeakEdge<Data, Weight>> const &edges() const { return edges_; }

    WeakEdge<Data, Weight> edge(int index) { return edges_.at(index); }

    WeakEdge<Data, Weight> const edge(int index) const { return edges_.at(index); }

    void add_edge(SharedEdge<Data, Weight> edge) { edges_.push_back(edge); }

    void remove_edge(SharedEdge<Data, Weight> edge) {
        for (auto it = edges_.begin(); it != edges_.end(); it++) {
            if (*it->lock() == *edge) {
                edges_.erase(it);
                return;
            }
        }
    }

    Data &data() { return data_; }

    Data const &data() const { return data_; }

  private:
    size_t serial_id_{0};

    std::vector<WeakEdge<Data, Weight>> edges_;

    Data data_;
};

template <typename Weight>
struct Vertex<void, Weight> {
  public:
    Vertex() : serial_id_{0}, edges_() {}

    template <typename... Args>
    Vertex(size_t serial_id) : serial_id_{serial_id}, edges_() {}

    Vertex(Vertex const &other) : serial_id_{other.id()}, edges_() {}

    virtual ~Vertex() { edges_.clear(); }

    SharedVertex<void, Weight> clone() const { return std::make_shared<Vertex<void, Weight>>(id()); }

    bool operator==(Vertex const &other) const { return id() == other.id(); }

    size_t id() const { return serial_id_; }

    void set_id(size_t new_id) { serial_id_ = new_id; }

    std::vector<WeakEdge<void, Weight>> &edges() { return edges_; }

    std::vector<WeakEdge<void, Weight>> const &edges() const { return edges_; }

    WeakEdge<void, Weight> edge(int index) { return edges_.at(index); }

    WeakEdge<void, Weight> const edge(int index) const { return edges_.at(index); }

    void add_edge(SharedEdge<void, Weight> edge) { edges_.push_back(edge); }

    void remove_edge(SharedEdge<void, Weight> edge) {
        for (auto it = edges_.begin(); it != edges_.end(); it++) {
            if (*it->lock() == *edge) {
                edges_.erase(it);
                return;
            }
        }
    }

  private:
    size_t serial_id_{0};

    std::vector<WeakEdge<void, Weight>> edges_;
};

template <typename Data, typename Weight>
struct Edge {
  public:
    Edge() noexcept : start_(), end_(), serial_id_{0}, is_directed_{true}, weight_() {}

    template <typename... Args>
    Edge(size_t serial_id, SharedVertex<Data, Weight> const &start, SharedVertex<Data, Weight> const &end, bool is_directed, Args &&...args)
        : start_(start), end_(end), serial_id_{serial_id}, is_directed_{is_directed}, weight_(std::forward<Args>(args)...) {}
    Edge(Edge const &other) noexcept
        : start_(other.start()), end_(other.end()), serial_id_{other.id()}, is_directed_{other.is_directed()}, weight_(other.weight()) {}

    virtual ~Edge() {
        start_.reset();
        end_.reset();
    }

    bool operator==(Edge const &other) const { return id() == other.id(); }

    bool is_traversable(Vertex<Data, Weight> const &from) const {
        if (*start_.lock() == from) {
            return true;
        } else if (!is_directed() && *end_.lock() == from) {
            return true;
        }
        return false;
    }

    bool is_directed() const { return is_directed_; }

    void set_directed(bool new_directed) { is_directed_ = new_directed; }

    void swap_direction() { std::swap(start_, end_); }

    size_t id() const { return serial_id_; }

    void set_id(size_t new_id) { serial_id_ = new_id; }

    WeakVertex<Data, Weight> traverse(Vertex<Data, Weight> const &from) const {
        if (*start_.lock() == from) {
            return end();
        } else if (!is_directed() && *end_.lock() == from) {
            return start();
        }
        if (is_directed() && *end_.lock() == from) {
            EINSUMS_THROW_EXCEPTION(bad_logic,
                                    "Could not follow edge! Edge is directed, and the code is trying to traverse the wrong direction!");
        } else {
            EINSUMS_THROW_EXCEPTION(bad_logic, "Could not follow edge! The starting node is not part of the edge!");
        }
    }

    WeakVertex<Data, Weight> start() { return start_; }

    WeakVertex<Data, Weight> end() { return end_; }

    WeakVertex<Data, Weight> const start() const { return start_; }

    WeakVertex<Data, Weight> const end() const { return end_; }

    SharedEdge<Data, Weight> clone() const { return std::make_shared<Edge<Data, Weight>>(*this); }

    Weight &weight() { return weight_; }

    Weight const &weight() const { return weight_; }

  private:
    WeakVertex<Data, Weight> start_, end_;

    bool   is_directed_{true};
    size_t serial_id_{0};

    Weight weight_;
};

template <typename Data>
struct Edge<Data, void> {
  public:
    Edge() noexcept : start_(), end_(), serial_id_{0}, is_directed_{true} {}

    template <typename... Args>
    Edge(size_t serial_id, SharedVertex<Data, void> const &start, SharedVertex<Data, void> const &end, bool is_directed)
        : start_(start), end_(end), serial_id_{serial_id}, is_directed_{is_directed} {}
    Edge(Edge const &other) noexcept
        : start_(other.start()), end_(other.end()), serial_id_{other.id()}, is_directed_{other.is_directed()} {}

    virtual ~Edge() {
        start_.reset();
        end_.reset();
    }

    bool operator==(Edge const &other) const { return id() == other.id(); }

    bool is_traversable(Vertex<Data, void> const &from) const {
        if (*start_.lock() == from) {
            return true;
        } else if (!is_directed() && *end_.lock() == from) {
            return true;
        }
        return false;
    }

    bool is_directed() const { return is_directed_; }

    void set_directed(bool new_directed) { is_directed_ = new_directed; }

    void swap_direction() { std::swap(start_, end_); }

    size_t id() const { return serial_id_; }

    void set_id(size_t new_id) { serial_id_ = new_id; }

    WeakVertex<Data, void> traverse(Vertex<Data, void> const &from) const {
        if (*start_.lock() == from) {
            return end();
        } else if (!is_directed() && *end_.lock() == from) {
            return start();
        }
        if (is_directed() && *end_.lock() == from) {
            EINSUMS_THROW_EXCEPTION(bad_logic,
                                    "Could not follow edge! Edge is directed, and the code is trying to traverse the wrong direction!");
        } else {
            EINSUMS_THROW_EXCEPTION(bad_logic, "Could not follow edge! The starting node is not part of the edge!");
        }
    }

    WeakVertex<Data, void> start() { return start_; }

    WeakVertex<Data, void> end() { return end_; }

    WeakVertex<Data, void> const start() const { return start_; }

    WeakVertex<Data, void> const end() const { return end_; }

    SharedEdge<Data, void> clone() const { return std::make_shared<Edge<Data, void>>(*this); }

  private:
    WeakVertex<Data, void> start_, end_;

    bool   is_directed_{true};
    size_t serial_id_{0};
};

template <typename Data, typename Weight>
struct EdgeCompare {
    bool operator()(std::shared_ptr<Edge<Data, Weight>> const &a, std::shared_ptr<Edge<Data, Weight>> const &b) const {
        return a->weight() < b->weight();
    }
};

template <typename Data, typename Weight>
struct Graph : design_pats::Lockable<std::mutex> {
  public:
    Graph() : vertices_(), edges_(), vertex_id_{0}, edge_id_{0} {}

    Graph(Graph const &other) : vertices_(), edges_(), vertex_id_{other.vertex_id_}, edge_id_{other.edge_id_} {
        for (auto vertex : other.vertices()) {
            vertices_.emplace_back(vertex->clone());
        }

        for (auto edge : other.edges()) {
            auto new_edge     = edge->clone();
            new_edge->start() = vertex_by_id(edge->start().lock()->id());
            new_edge->end()   = vertex_by_id(edge->end().lock()->id());

            edges_.push_back(new_edge);
        }
    }

    virtual ~Graph() {
        edges_.clear();
        vertices_.clear();
    }

    std::vector<SharedVertex<Data, Weight>> &vertices() { return vertices_; }

    std::vector<SharedVertex<Data, Weight>> const &vertices() const { return vertices_; }

    std::vector<SharedEdge<Data, Weight>> &edges() { return edges_; }

    std::vector<SharedEdge<Data, Weight>> const &edges() const { return edges_; }

    SharedVertex<Data, Weight> vertex(size_t index) { return vertices_.at(index); }

    SharedVertex<Data, Weight> const vertex(size_t index) const { return vertices_.at(index); }

    SharedEdge<Data, Weight> edge(size_t index) { return edges_.at(index); }

    SharedEdge<Data, Weight> const edge(size_t index) const { return edges_.at(index); }

    SharedVertex<Data, Weight> vertex_by_id(size_t id) {
        for (auto vertex : vertices_) {
            if (vertex->id() == id) {
                return vertex;
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find vertex with the given ID!");
    }

    SharedVertex<Data, Weight> const vertex_by_id(size_t id) const {
        for (auto vertex : vertices_) {
            if (vertex->id() == id) {
                return vertex;
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find vertex with the given ID!");
    }

    SharedEdge<Data, Weight> edge_by_id(size_t id) {
        for (auto edge : edges_) {
            if (edge->id() == id) {
                return edge;
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find edge with the given ID!");
    }
    SharedEdge<Data, Weight> const edge_by_id(size_t id) const {
        for (auto edge : edges_) {
            if (edge->id() == id) {
                return edge;
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find edge with the given ID!");
    }

    bool has_vertex(size_t id) const {
        for (auto vertex : vertices_) {
            if (vertex->id() == id) {
                return true;
            }
        }
        return false;
    }

    bool has_vertex(Vertex<Data, Weight> const &other) const {
        for (auto vertex : vertices_) {
            if (*vertex == other) {
                return true;
            }
        }
        return false;
    }

    bool has_vertex(SharedVertex<Data, Weight> other) const {
        for (auto vertex : vertices_) {
            if (vertex == other) {
                return true;
            }
        }
        return false;
    }

    bool has_edge(size_t id) const {
        for (auto edge : edges_) {
            if (edge->id() == id) {
                return true;
            }
        }
        return false;
    }

    bool has_edge(Edge<Data, Weight> const &other) const {
        for (auto edge : edges_) {
            if (*edge == other) {
                return true;
            }
        }
        return false;
    }
    bool has_edge(SharedEdge<Data, Weight> other) const {
        for (auto edge : edges_) {
            if (edge == other) {
                return true;
            }
        }
        return false;
    }

    size_t num_vertices() const { return vertices_.size(); }

    size_t num_edges() const { return edges_.size(); }

    void push_vertex(SharedVertex<Data, Weight> vertex) {
        auto lock = std::lock_guard(*this);
        vertices_.push_back(vertex);

        if (vertex->id() > vertex_id_) {
            vertex_id_ = vertex->id() + 1;
        }
    }

    void push_edge(SharedEdge<Data, Weight> edge) {
        auto lock = std::lock_guard(*this);
        edges_.push_back(edge);
        if (edge->id() > edge_id_) {
            edge_id_ = edge->id() + 1;
        }
    }

    template <typename... Args>
    SharedVertex<Data, Weight> emplace_vertex(Args &&...args) {
        auto lock = std::lock_guard(*this);

        auto out = std::make_shared<Vertex<Data, Weight>>(vertex_id_, std::forward<Args>(args)...);

        vertices_.push_back(out);

        vertex_id_++;

        return out;
    }

    template <typename... Args>
    SharedEdge<Data, Weight> emplace_edge(Args &&...args) {
        auto lock = std::lock_guard(*this);

        auto out = std::make_shared<Edge<Data, Weight>>(edge_id_, std::forward<Args>(args)...);

        out->start().lock()->add_edge(out);

        if (!out->is_directed()) {
            out->end().lock()->add_edge(out);
        }

        edges_.push_back(out);

        edge_id_++;

        return out;
    }

    void pop_vertex(size_t index) {
        auto lock = std::lock_guard(*this);
        vertices_.erase(std::next(vertices_.begin(), index));
    }
    void pop_edge(size_t index) {
        auto lock = std::lock_guard(*this);
        edges_.erase(std::next(edges_.begin(), index));
    }

    Graph clone() const { return Graph(*this); }

  private:
    std::vector<SharedVertex<Data, Weight>> vertices_;
    std::vector<SharedEdge<Data, Weight>>   edges_;

    size_t vertex_id_{0}, edge_id_{0};
};

#ifndef EINSUMS_WINDOWS
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Edge<void, void>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Edge<void, int>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Edge<void, double>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Vertex<void, void>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Vertex<void, int>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Vertex<void, double>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Graph<void, void>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Graph<void, int>;
extern template struct EINSUMS_EXPERIMENTAL_EXPORT Graph<void, double>;
#endif

} // namespace graph
} // namespace einsums