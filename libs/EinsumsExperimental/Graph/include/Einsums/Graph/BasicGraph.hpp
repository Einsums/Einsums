//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <deque>
#include <memory>
#include "Einsums/TypeSupport/Lockable.hpp"

namespace einsums {

namespace graph {

template <typename EdgeWeight>
struct BasicEdge;
template <>
struct BasicEdge<void>;

template <typename EdgeWeight>
struct BasicNode {
  public:
    BasicNode(size_t id) : edges_(), serial_id_{id} {}
    BasicNode(BasicNode<EdgeWeight> const &) = delete;

    virtual ~BasicNode() { edges_.clear(); }

    void add_edge(std::shared_ptr<BasicEdge<EdgeWeight>> new_edge) { edges_.push_back(new_edge); }

    std::deque<std::shared_ptr<BasicEdge<EdgeWeight>>> &get_edges() { return edges_; }

    virtual BasicNode<EdgeWeight> create_copy() const {
        return BasicNode(serial_id_);
    }

    virtual bool operator==(BasicNode<EdgeWeight> const &other) const {
        return other.id() == this->id();
    }

    size_t id() const {
        return serial_id_;
    }

  private:
    std::deque<std::shared_ptr<BasicEdge<EdgeWeight>>> edges_;
    size_t serial_id_;
};

template <typename EdgeWeight>
struct BasicEdge {
  public:
    BasicEdge(std::shared_ptr<BasicNode<EdgeWeight>> start, std::shared_ptr<BasicNode<EdgeWeight>> end, EdgeWeight weight)
        : start_(start), end_(end), weight_{weight} {}

    virtual ~BasicEdge() = default;

    virtual std::weak_ptr<BasicNode<EdgeWeight>> const traverse(std::shared_ptr<BasicNode<EdgeWeight>> input) const = 0;
    virtual std::weak_ptr<BasicNode<EdgeWeight>>       traverse(std::shared_ptr<BasicNode<EdgeWeight>> input)       = 0;

    virtual bool is_traversable(std::shared_ptr<BasicNode<EdgeWeight>> input) const = 0;

    std::weak_ptr<BasicNode<EdgeWeight>> const get_start() const { return start_; }

    std::weak_ptr<BasicNode<EdgeWeight>> const get_end() const { return end_; }

    std::weak_ptr<BasicNode<EdgeWeight>> get_start() { return start_; }

    std::weak_ptr<BasicNode<EdgeWeight>> get_end() { return end_; }

    EdgeWeight const &weight() const { return weight_; }

    EdgeWeight &weight() { return weight_; }

  protected:
    std::weak_ptr<BasicNode<EdgeWeight>> start_, end_;
    EdgeWeight                           weight_;
};

template <typename EdgeWeight>
struct BasicDirectedEdge : public BasicEdge<EdgeWeight> {
  public:
    using BasicEdge<EdgeWeight>::BasicEdge;

    virtual ~BasicDirectedEdge() = default;

    std::weak_ptr<BasicNode<EdgeWeight>> const traverse(std::shared_ptr<BasicNode<EdgeWeight>> input) const override {
        auto start_lock = this->get_start().lock();

        if (start_lock.get() == input.get()) {
            return this->get_end();
        }
        return std::weak_ptr<BasicNode<EdgeWeight>>();
    }
    std::weak_ptr<BasicNode<EdgeWeight>> traverse(std::shared_ptr<BasicNode<EdgeWeight>> input) override {
        auto start_lock = this->get_start().lock();

        if (start_lock.get() == input.get()) {
            return this->get_end();
        }
        return std::weak_ptr<BasicNode<EdgeWeight>>();
    }

    bool is_traversable(std::shared_ptr<BasicNode<EdgeWeight>> input) const override {
        auto start_lock = this->get_start().lock();

        if (start_lock.get() == input.get()) {
            return true;
        }
        return false;
    }
};

template <typename EdgeWeight>
struct BasicUndirectedEdge : public BasicEdge<EdgeWeight> {
  public:
    using BasicEdge<EdgeWeight>::BasicEdge;

    virtual ~BasicUndirectedEdge() = default;

    std::weak_ptr<BasicNode<EdgeWeight>> const traverse(std::shared_ptr<BasicNode<EdgeWeight>> input) const override {
        auto start_lock = this->get_start().lock();

        if (start_lock.get() == input.get()) {
            return this->get_end();
        }
        return std::weak_ptr<BasicNode<EdgeWeight>>();
    }
    std::weak_ptr<BasicNode<EdgeWeight>> traverse(std::shared_ptr<BasicNode<EdgeWeight>> input) override {
        if (this->get_start().lock().get() == input.get()) {
            return this->get_end();
        }
        if (this->get_end().lock().get() == input.get()) {
            return this->get_start();
        }
        return std::weak_ptr<BasicNode<EdgeWeight>>();
    }

    bool is_traversable(std::shared_ptr<BasicNode<EdgeWeight>> input) const override {
        if (this->get_start().lock().get() == input.get()) {
            return true;
        }
        if (this->get_end().lock().get() == input.get()) {
            return true;
        }
        return false;
    }
};

template<typename EdgeWeight>
struct EdgeCompare {
    public:
    bool operator()(std::shared_ptr<BasicEdge<EdgeWeight>> const a, std::shared_ptr<BasicEdge<EdgeWeight>> const b) const {
        return a->weight < b->weight;
    }
};

template <typename EdgeWeight>
struct BasicGraph : public design_pats::Lockable<std::mutex> {
  public:
    BasicGraph() : nodes_(), edges_() {}

    virtual ~BasicGraph() {
        edges_.clear();
        nodes_.clear();
    }

    void add_node(std::shared_ptr<BasicNode<EdgeWeight>> node) { nodes_.push_back(node); }

    void add_edge(std::shared_ptr<BasicEdge<EdgeWeight>> edge) {
        edges_.push_back(edge);
        for (auto &node : nodes_) {
            if (edge->is_traversable(node)) {
                node->add_edge(edge);
            }
        }
    }

    std::deque<std::shared_ptr<BasicNode<EdgeWeight>>> &get_nodes() { return nodes_; }

    std::deque<std::shared_ptr<BasicNode<EdgeWeight>>> const &get_nodes() const { return nodes_; }

    std::deque<std::shared_ptr<BasicEdge<EdgeWeight>>> &get_edges() { return edges_; }

    std::deque<std::shared_ptr<BasicEdge<EdgeWeight>>> const &get_edges() const { return edges_; }

  private:
    std::deque<std::shared_ptr<BasicNode<EdgeWeight>>> nodes_;
    std::deque<std::shared_ptr<BasicEdge<EdgeWeight>>> edges_;
};

template <>
struct EINSUMS_EXPERIMENTAL_EXPORT BasicNode<void> {
  public:
    BasicNode(size_t id);
    BasicNode(BasicNode<void> const &) = delete;

    virtual ~BasicNode();

    void add_edge(std::shared_ptr<BasicEdge<void>> new_edge);

    std::deque<std::shared_ptr<BasicEdge<void>>> &get_edges();

    virtual bool operator==(BasicNode<void> const &other) const;

    size_t id() const;

  private:
    std::deque<std::shared_ptr<BasicEdge<void>>> edges_;
    size_t serial_id_;
};

template <>
struct EINSUMS_EXPERIMENTAL_EXPORT BasicEdge<void> {
  public:
    BasicEdge(std::shared_ptr<BasicNode<void>> start, std::shared_ptr<BasicNode<void>> end);

    virtual ~BasicEdge() = default;

    virtual std::weak_ptr<BasicNode<void>> const traverse(std::shared_ptr<BasicNode<void>> input) const = 0;
    virtual std::weak_ptr<BasicNode<void>>       traverse(std::shared_ptr<BasicNode<void>> input)       = 0;

    virtual bool is_traversable(std::shared_ptr<BasicNode<void>> input) const = 0;

    std::weak_ptr<BasicNode<void>> const get_start() const;

    std::weak_ptr<BasicNode<void>> const get_end() const;

    std::weak_ptr<BasicNode<void>> get_start();

    std::weak_ptr<BasicNode<void>> get_end();

  protected:
    std::weak_ptr<BasicNode<void>> start_, end_;
};

template <>
struct EINSUMS_EXPERIMENTAL_EXPORT BasicDirectedEdge<void> : public BasicEdge<void> {
  public:
    using BasicEdge<void>::BasicEdge;

    virtual ~BasicDirectedEdge() = default;

    std::weak_ptr<BasicNode<void>> const traverse(std::shared_ptr<BasicNode<void>> input) const override;
    std::weak_ptr<BasicNode<void>>       traverse(std::shared_ptr<BasicNode<void>> input) override;

    bool is_traversable(std::shared_ptr<BasicNode<void>> input) const override;
};

template <>
struct EINSUMS_EXPERIMENTAL_EXPORT BasicUndirectedEdge<void> : public BasicEdge<void> {
  public:
    using BasicEdge<void>::BasicEdge;

    virtual ~BasicUndirectedEdge() = default;

    std::weak_ptr<BasicNode<void>> const traverse(std::shared_ptr<BasicNode<void>> input) const override;
    std::weak_ptr<BasicNode<void>>       traverse(std::shared_ptr<BasicNode<void>> input) override;

    bool is_traversable(std::shared_ptr<BasicNode<void>> input) const override;
};

template <>
struct EINSUMS_EXPERIMENTAL_EXPORT BasicGraph<void> : public design_pats::Lockable<std::mutex> {
  public:
    BasicGraph();

    virtual ~BasicGraph();

    size_t pop_serial_id();

    void add_node(std::shared_ptr<BasicNode<void>> node);

    void add_edge(std::shared_ptr<BasicEdge<void>> edge);

    std::deque<std::shared_ptr<BasicNode<void>>> &get_nodes();

    std::deque<std::shared_ptr<BasicNode<void>>> const &get_nodes() const;

    std::deque<std::shared_ptr<BasicEdge<void>>> &get_edges();

    std::deque<std::shared_ptr<BasicEdge<void>>> const &get_edges() const;

  private:
    std::deque<std::shared_ptr<BasicNode<void>>> nodes_;
    std::deque<std::shared_ptr<BasicEdge<void>>> edges_;
    size_t serial_id_;
};

template <typename EdgeWeight>
using SharedEdge = typename std::shared_ptr<BasicEdge<EdgeWeight>>;
template <typename EdgeWeight>
using SharedNode = typename std::shared_ptr<BasicNode<EdgeWeight>>;
template <typename EdgeWeight>
using SharedUndirectedEdge = typename std::shared_ptr<BasicUndirectedEdge<EdgeWeight>>;
template <typename EdgeWeight>
using SharedDirectedEdge = typename std::shared_ptr<BasicDirectedEdge<EdgeWeight>>;
template <typename EdgeWeight>
using SharedGraph = typename std::shared_ptr<BasicGraph<EdgeWeight>>;

#ifndef _MSC_VER
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicNode<int>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicEdge<int>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicUndirectedEdge<int>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicDirectedEdge<int>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicGraph<int>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicNode<double>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicEdge<double>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicUndirectedEdge<double>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicDirectedEdge<double>;
extern template class EINSUMS_EXPERIMENTAL_EXPORT BasicGraph<double>;
#endif

} // namespace graph

} // namespace einsums