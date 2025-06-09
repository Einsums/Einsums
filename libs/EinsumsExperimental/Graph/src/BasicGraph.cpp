#include <Einsums/Graph/BasicGraph.hpp>

namespace einsums {
namespace graph {

#ifndef _MSC_VER
template class BasicNode<int>;
template class BasicEdge<int>;
template class BasicUndirectedEdge<int>;
template class BasicDirectedEdge<int>;
template class BasicGraph<int>;
template class BasicNode<double>;
template class BasicEdge<double>;
template class BasicUndirectedEdge<double>;
template class BasicDirectedEdge<double>;
template class BasicGraph<double>;
#endif

BasicNode<void>::BasicNode(size_t id) : edges_(), serial_id_{id} {
}

BasicNode<void>::~BasicNode() {
    edges_.clear();
}

void BasicNode<void>::add_edge(std::shared_ptr<BasicEdge<void>> new_edge) {
    edges_.push_back(new_edge);
}

std::deque<std::shared_ptr<BasicEdge<void>>> &BasicNode<void>::get_edges() {
    return edges_;
}

bool BasicNode<void>::operator==(BasicNode<void> const &other) const {
    return other.id() == this->id();
}

size_t BasicNode<void>::id() const {
    return serial_id_;
}

BasicEdge<void>::BasicEdge(std::shared_ptr<BasicNode<void>> start, std::shared_ptr<BasicNode<void>> end) : start_(start), end_(end) {
}

std::weak_ptr<BasicNode<void>> const BasicEdge<void>::get_start() const {
    return start_;
}

std::weak_ptr<BasicNode<void>> const BasicEdge<void>::get_end() const {
    return end_;
}

std::weak_ptr<BasicNode<void>> BasicEdge<void>::get_start() {
    return start_;
}

std::weak_ptr<BasicNode<void>> BasicEdge<void>::get_end() {
    return end_;
}

std::weak_ptr<BasicNode<void>> const BasicDirectedEdge<void>::traverse(std::shared_ptr<BasicNode<void>> input) const {
    auto start_lock = start_.lock();

    if (start_lock.get() == input.get()) {
        return get_end();
    }
    return std::weak_ptr<BasicNode<void>>();
}
std::weak_ptr<BasicNode<void>> BasicDirectedEdge<void>::traverse(std::shared_ptr<BasicNode<void>> input) {
    auto start_lock = start_.lock();

    if (start_lock.get() == input.get()) {
        return get_end();
    }
    return std::weak_ptr<BasicNode<void>>();
}

bool BasicDirectedEdge<void>::is_traversable(std::shared_ptr<BasicNode<void>> input) const {
    auto start_lock = start_.lock();

    if (start_lock.get() == input.get()) {
        return true;
    }
    return false;
}

std::weak_ptr<BasicNode<void>> const BasicUndirectedEdge<void>::traverse(std::shared_ptr<BasicNode<void>> input) const {
    auto start_lock = get_start().lock();

    if (start_lock.get() == input.get()) {
        return get_end();
    }
    return std::weak_ptr<BasicNode<void>>();
}
std::weak_ptr<BasicNode<void>> BasicUndirectedEdge<void>::traverse(std::shared_ptr<BasicNode<void>> input) {
    if (get_start().lock().get() == input.get()) {
        return get_end();
    }
    if (get_end().lock().get() == input.get()) {
        return get_start();
    }
    return std::weak_ptr<BasicNode<void>>();
}

bool BasicUndirectedEdge<void>::is_traversable(std::shared_ptr<BasicNode<void>> input) const {
    if (get_start().lock().get() == input.get()) {
        return true;
    }
    if (get_end().lock().get() == input.get()) {
        return true;
    }
    return false;
}

BasicGraph<void>::BasicGraph() : nodes_(), edges_() {
}

BasicGraph<void>::~BasicGraph() {
    edges_.clear();
    nodes_.clear();
}

size_t BasicGraph<void>::pop_serial_id() {
    auto   guard = std::lock_guard(*this);
    size_t ret   = serial_id_;

    serial_id_++;
    return ret;
}

void BasicGraph<void>::add_node(std::shared_ptr<BasicNode<void>> node) {
    auto guard = std::lock_guard(*this);
    nodes_.push_back(node);

    if(node->id() >= serial_id_) {
        serial_id_ = node->id() + 1;
    } 
}

void BasicGraph<void>::add_edge(std::shared_ptr<BasicEdge<void>> edge) {
    auto guard = std::lock_guard(*this);
    edges_.push_back(edge);

    for (auto &node : nodes_) {
        if (edge->is_traversable(node)) {
            node->add_edge(edge);
        }
    }
}

std::deque<std::shared_ptr<BasicNode<void>>> &BasicGraph<void>::get_nodes() {
    return nodes_;
}

std::deque<std::shared_ptr<BasicNode<void>>> const &BasicGraph<void>::get_nodes() const {
    return nodes_;
}

std::deque<std::shared_ptr<BasicEdge<void>>> &BasicGraph<void>::get_edges() {
    return edges_;
}

std::deque<std::shared_ptr<BasicEdge<void>>> const &BasicGraph<void>::get_edges() const {
    return edges_;
}

} // namespace graph
} // namespace einsums