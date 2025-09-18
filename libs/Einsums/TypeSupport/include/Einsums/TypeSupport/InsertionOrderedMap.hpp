//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <unordered_map>
#include <vector>

namespace einsums {

template <class K, class V, class Hash = std::hash<K>, class Eq = std::equal_to<K>>
class InsertionOrderedMap {
  public:
    using map_type      = std::unordered_map<K, V, Hash, Eq>;
    using value_type    = typename map_type::value_type; // pair<const K, V>
    using size_type     = typename map_type::size_type;
    using map_iterator  = typename map_type::iterator;
    using map_citerator = typename map_type::const_iterator;

  private:
    // We iterate over order_ (vector of iterators into map_) to preserve insertion order.
    std::vector<map_iterator> order_;
    map_type                  map_;

    // (Optional) accelerate find->order position; skip for brevity/clarity.
    // std::unordered_map<K, size_t, Hash, Eq> pos_;

  public:
    InsertionOrderedMap() = default;

    // ----- Iterator types that yield references into map_ -----
    class iterator {
      public:
        using vec_iter          = typename std::vector<map_iterator>::const_iterator;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::bidirectional_iterator_tag;
        using reference         = value_type &;
        using pointer           = value_type *;

        iterator() = default;
        iterator(vec_iter it) : it_(it) {}

        reference operator*() const { return **it_; }
        pointer   operator->() const { return std::addressof(**it_); }

        iterator &operator++() {
            ++it_;
            return *this;
        }
        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }
        iterator &operator--() {
            --it_;
            return *this;
        }
        iterator operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        friend bool operator==(iterator const &a, iterator const &b) { return a.it_ == b.it_; }
        friend bool operator!=(iterator const &a, iterator const &b) { return !(a == b); }
        // expose order iterator for internal construction
        vec_iter raw() const { return it_; }

      private:
        vec_iter it_{};
    };

    class const_iterator {
      public:
        using vec_iter          = typename std::vector<map_iterator>::const_iterator;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::bidirectional_iterator_tag;
        using reference         = value_type const &;
        using pointer           = value_type const *;

        const_iterator() = default;
        const_iterator(vec_iter it) : it_(it) {}
        const_iterator(iterator const &it) : it_(it.raw()) {}

        reference operator*() const { return **it_; }
        pointer   operator->() const { return std::addressof(**it_); }

        const_iterator &operator++() {
            ++it_;
            return *this;
        }
        const_iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }
        const_iterator &operator--() {
            --it_;
            return *this;
        }
        const_iterator operator--(int) {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        friend bool operator==(const_iterator const &a, const_iterator const &b) { return a.it_ == b.it_; }
        friend bool operator!=(const_iterator const &a, const_iterator const &b) { return !(a == b); }

      private:
        vec_iter it_{};
    };

    // ----- Capacity -----
    [[nodiscard]] bool empty() const noexcept { return map_.empty(); }
    size_type          size() const noexcept { return map_.size(); }

    // ----- Iteration (in insertion order) -----
    iterator       begin() { return iterator{order_.begin()}; }
    iterator       end() { return iterator{order_.end()}; }
    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }
    const_iterator cbegin() const { return const_iterator{order_.begin()}; }
    const_iterator cend() const { return const_iterator{order_.end()}; }

    // ----- Lookup -----
    iterator find(K const &key) {
        auto mit = map_.find(key);
        if (mit == map_.end())
            return end();
        // locate iterator in order_ (O(n)); for O(1) maintain a pos_ index.
        auto vit = std::find_if(order_.begin(), order_.end(), [&](map_iterator const &it) { return it == mit; });
        if (vit == order_.end())
            return end(); // should not happen
        return iterator{vit};
    }

    const_iterator find(K const &key) const {
        auto mit = map_.find(key);
        if (mit == map_.end())
            return cend();
        auto vit = std::find_if(order_.begin(), order_.end(), [&](map_iterator const &it) { return it == mit; });
        if (vit == order_.end())
            return cend();
        return const_iterator{vit};
    }

    // ----- Insertion -----
    // insert/emplace preserve first-in insertion order
    std::pair<iterator, bool> insert(value_type const &kv) {
        auto [mit, inserted] = map_.insert(kv);
        if (inserted)
            order_.push_back(mit);
        // find order position (last if inserted)
        auto vit = inserted ? std::prev(order_.end())
                            : std::find_if(order_.begin(), order_.end(), [&](map_iterator const &it) { return it == mit; });
        return {iterator{vit}, inserted};
    }

    template <class... Args>
    std::pair<iterator, bool> emplace(Args &&...args) {
        auto [mit, inserted] = map_.emplace(std::forward<Args>(args)...);
        if (inserted)
            order_.push_back(mit);
        auto vit = inserted ? std::prev(order_.end())
                            : std::find_if(order_.begin(), order_.end(), [&](map_iterator const &it) { return it == mit; });
        return {iterator{vit}, inserted};
    }

    // like operator[]: inserts default if missing and returns reference
    V &operator[](K const &key) {
        auto [mit, inserted] = map_.try_emplace(key);
        if (inserted)
            order_.push_back(mit);
        return mit->second;
    }

    V       &at(K const &key) { return map_.at(key); }
    V const &at(K const &key) const { return map_.at(key); }

    // (optional) erase by key; keeps order_ in sync
    size_type erase(K const &key) {
        auto mit = map_.find(key);
        if (mit == map_.end())
            return 0;
        auto vit = std::find_if(order_.begin(), order_.end(), [&](map_iterator const &it) { return it == mit; });
        if (vit != order_.end())
            order_.erase(vit);
        map_.erase(mit);
        return 1;
    }
};

} // namespace einsums
