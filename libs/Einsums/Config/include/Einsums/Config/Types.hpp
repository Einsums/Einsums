//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace einsums {

template <typename Value>
class ConfigObserver;

template <typename Value>
class ConfigMap : public std::enable_shared_from_this<ConfigMap<Value>> {
  private:
    class PrivateType {
      public:
        explicit PrivateType() = default;
    };

  public:
    using MappingType          = std::unordered_map<std::string, Value>;
    using key_type             = typename MappingType::key_type;
    using mapped_type          = typename MappingType::mapped_type;
    using value_type           = typename MappingType::value_type;
    using size_type            = typename MappingType::size_type;
    using difference_type      = typename MappingType::difference_type;
    using hasher               = typename MappingType::hasher;
    using key_equal            = typename MappingType::key_equal;
    using allocator_type       = typename MappingType::allocator_type;
    using reference            = typename MappingType::reference;
    using const_reference      = typename MappingType::const_reference;
    using pointer              = typename MappingType::pointer;
    using const_pointer        = typename MappingType::const_pointer;
    using iterator             = typename MappingType::iterator;
    using const_iterator       = typename MappingType::const_iterator;
    using local_iterator       = typename MappingType::local_iterator;
    using const_local_iterator = typename MappingType::const_local_iterator;
    using node_type            = typename MappingType::node_type;
    using insert_return_type   = typename MappingType::insert_return_type;
    using Key                  = std::string;

    ConfigMap(PrivateType) : map_() {}

    void attach(std::shared_ptr<ConfigObserver<Value>> obs) { observers_.emplace_back(obs); }

    void detach(std::shared_ptr<ConfigObserver<Value>> obs) {
        auto curr_iter = this->observers_.begin();

        while (curr_iter != this->observers_.end()) {
            auto temp_iter = curr_iter;
            curr_iter      = std::next(temp_iter);
            // Clear expired observers and the requested observer.
            if (temp_iter->expired() || temp_iter->lock() == obs) {
                this->observers_.erase(temp_iter);
            }
        }
    }

    void notify() {
        for (auto &obs : this->observers_) {
            obs.lock()->update(this->shared_from_this());
        }
    }

    static std::shared_ptr<ConfigMap<Value>> create() { return std::make_shared<ConfigMap<Value>>(PrivateType()); }

    // Map wrapping.

    inline allocator_type get_allocator() const noexcept { return map_.get_allocator(); }

    inline iterator begin() noexcept { return map_.begin(); }

    inline const_iterator begin() const noexcept { return map_.begin(); }

    inline const_iterator cbegin() const noexcept { return map_.cbegin(); }

    inline iterator end() noexcept { return map_.end(); }

    inline const_iterator end() const noexcept { return map_.end(); }

    inline const_iterator cend() const noexcept { return map_.cend(); }

    inline bool empty() const noexcept { return map_.empty(); }

    inline size_type size() const noexcept { return map_.size(); }

    inline size_type max_size() const noexcept { return map_.max_size(); }

    inline void clear() noexcept { map_.clear(); }

    inline std::pair<iterator, bool> insert(value_type const &value) { return map_.insert(value); }

    inline std::pair<iterator, bool> insert(value_type &&value) { return map_.insert(std::forward<value_type>(value)); }

    template <class P>
    std::pair<iterator, bool> insert(P &&value) {
        return map_.insert(std::forward<P>(value));
    }

    inline iterator insert(const_iterator hint, value_type const &value) { return map_.insert(hint, value); }

    inline iterator insert(const_iterator hint, value_type &&value) { return map_.insert(hint, std::forward<value_type>(value)); }

    template <class P>
    iterator insert(const_iterator hint, P &&value) {
        return map_.insert(hint, std::forward<P>(value));
    }

    template <class InputIt>
    void insert(InputIt first, InputIt last) {
        map_.insert(first, last);
    }

    inline void insert(std::initializer_list<value_type> ilist) { map_.insert(ilist); }

    inline insert_return_type insert(node_type &&nh) { return map_.insert(std::forward<node_type>(nh)); }

    inline iterator insert(const_iterator hint, node_type &&nh) { return map_.insert(hint, std::forward<node_type>(nh)); }

    template <class M>
    std::pair<iterator, bool> insert_or_assign(Key const &key, M &&obj) {
        return map_.insert_or_assign(key, std::forward<M>(obj));
    }

    template <class M>
    std::pair<iterator, bool> insert_or_assign(Key &&key, M &&obj) {
        return map_.insert_or_assign(std::forward<Key>(key), std::forward<M>(obj));
    }

    template <class M>
    iterator insert_or_assign(const_iterator hint, Key const &key, M &&obj) {
        return map_.insert_or_assign(hint, key, std::forward<M>(obj));
    }

    template <class M>
    iterator insert_or_assign(const_iterator hint, Key &&key, M &&obj) {
        return map_.insert_or_assign(hint, std::forward<Key>(key), std::forward<M>(obj));
    }

    template <class... Args>
    std::pair<iterator, bool> emplace(Args &&...args) {
        return map_.emplace(std::forward<Args>(args)...);
    }

    template <class... Args>
    iterator emplace_hint(const_iterator hint, Args &&...args) {
        return map_.emplace_hint(hint, std::forward<Args>(args)...);
    }

    template <class... Args>
    std::pair<iterator, bool> try_emplace(Key const &key, Args &&...args) {
        return map_.try_emplace(key, std::forward<Args>(args)...);
    }

    template <class... Args>
    std::pair<iterator, bool> try_emplace(Key &&key, Args &&...args) {
        return map_.try_emplace(std::forward<Key>(key), std::forward<Args>(args)...);
    }

    template <class... Args>
    iterator try_emplace(const_iterator hint, Key const &key, Args &&...args) {
        return map_.try_emplace(hint, key, std::forward<Args>(args)...);
    }

    template <class... Args>
    iterator try_emplace(const_iterator hint, Key &&key, Args &&...args) {
        return map_.try_emplace(hint, std::forward<Key>(key), std::forward<Args>(args)...);
    }

    inline iterator erase(iterator pos) { return map_.erase(pos); }

    inline iterator erase(const_iterator pos) { return map_.erase(pos); }

    inline iterator erase(const_iterator first, const_iterator last) { return map_.erase(first, last); }

    inline size_type erase(Key const &key) { return map_.erase(key); }

    inline void swap(MappingType &other) noexcept(noexcept(std::allocator_traits<allocator_type>::is_always_equal::value &&
                                                           std::is_nothrow_swappable<hasher>::value &&
                                                           std::is_nothrow_swappable<key_equal>::value)) {
        map_.swap(other);
    }

    inline void swap(ConfigMap<Value> &other) noexcept(noexcept(std::allocator_traits<allocator_type>::is_always_equal::value &&
                                                                std::is_nothrow_swappable<hasher>::value &&
                                                                std::is_nothrow_swappable<key_equal>::value)) {
        map_.swap(other.map_);
    }

    inline node_type extract(const_iterator pos) { return map_.extract(pos); }

    inline node_type extract(Key const &key) { return map_.extract(key); }

    template <class H2, class P2>
    void merge(std::unordered_map<Key, Value, H2, P2, allocator_type> &other) {
        map_.merge(other);
    }

    template <class H2, class P2>
    void merge(std::unordered_map<Key, Value, H2, P2, allocator_type> &&other) {
        map_.merge(std::forward<std::unordered_map<Key, Value, H2, P2, allocator_type>>(other));
    }

    template <class H2, class P2>
    void merge(std::unordered_multimap<Key, Value, H2, P2, allocator_type> &other) {
        map_.merge(other);
    }

    template <class H2, class P2>
    void merge(std::unordered_multimap<Key, Value, H2, P2, allocator_type> &&other) {
        map_.merge(std::forward<std::unordered_multimap<Key, Value, H2, P2, allocator_type>>(other));
    }

    inline void merge(ConfigMap<Value> &other) { map_.merge(other.map_); }

    template <class H2, class P2>
    void merge(ConfigMap<Value> &&other) {
        map_.merge(std::forward<std::unordered_map<Key, Value, H2, P2, allocator_type>>(other._map));
    }

    inline Value &at(Key const &key) { return map_.at(key); }

    inline Value const &at(Key const &key) const { return map_.at(key); }

    inline Value &operator[](Key const &key) { return map_[key]; }

    inline Value const &operator[](Key const &key) const { return map_[key]; }

    inline size_type count(Key const &key) const { return map_.count(key); }

    template <class K>
    size_type count(K const &key) const {
        return map_.count(key);
    }

    inline iterator find(Key const &key) { return map_.find(key); }

    inline const_iterator find(Key const &key) const { return map_.find(key); }

    template <class K>
    iterator find(K const &key) {
        return map_.find(key);
    }

    template <class K>
    const_iterator find(K const &key) const {
        return map_.find(key);
    }

    inline bool contains(Key const &key) const { return map_.contains(key); }

    template <class K>
    bool contains(K const &key) const {
        return map_.contains(key);
    }

    inline std::pair<iterator, iterator> equal_range(Key const &key) { return map_.equal_range(key); }

    inline std::pair<const_iterator, const_iterator> equal_range(Key const &key) const { return map_.equal_range(key); }

    template <class K>
    std::pair<iterator, iterator> equal_range(K const &key) {
        return map_.equal_range(key);
    }

    template <class K>
    std::pair<const_iterator, const_iterator> equal_range(K const &key) const {
        return map_.equal_range(key);
    }

    inline local_iterator begin(size_type n) { return map_.begin(n); }

    inline const_local_iterator begin(size_type n) const { return map_.begin(n); }

    inline const_local_iterator cbegin(size_type n) const { return map_.cbegin(n); }

    inline local_iterator end(size_type n) { return map_.end(n); }

    inline const_local_iterator end(size_type n) const { return map_.end(n); }

    inline const_local_iterator cend(size_type n) const { return map_.cend(n); }

    inline size_type bucket_count() const { return map_.bucket_count(); }

    inline size_type max_bucket_count() const { return map_.max_bucket_count(); }

    inline size_type bucket_size(size_type n) const { return map_.bucket_size(n); }

    inline size_type bucket(Key const &key) const { return map_.bucket(key); }

    inline float load_factor() const { return map_.load_factor(); }

    inline float max_load_factor() const { return map_.max_load_factor(); }

    inline void max_load_factor(float ml) { map_.max_load_factor(ml); }

    inline void rehash(size_type count) { map_.rehash(count); }

    inline void reserve(size_type count) { map_.reserve(count); }

    inline hasher hash_function() const { return map_.hash_function(); }

    inline key_equal key_eq() const { return map_.key_eq(); }

    // Other methods.
    MappingType &get_map() { return map_; }

    MappingType const &get_map() const { return map_; }

    operator MappingType &() { return map_; }

    operator MappingType const &() const { return map_; }

  private:
    explicit ConfigMap() = default;

    std::unordered_map<std::string, Value> map_;

    std::list<std::weak_ptr<ConfigObserver<Value>>> observers_;

    friend class std::shared_ptr<ConfigObserver<Value>>;
};

template <typename Value>
using SharedConfigMap = std::shared_ptr<ConfigMap<Value>>;
// using SharedInfoMap = std::shared_ptr<InfoMap>;

template <typename Value>
class ConfigObserver {
  public:
    virtual ~ConfigObserver() = default;

    virtual void update(SharedConfigMap<Value>) = 0;
};

class EINSUMS_EXPORT GlobalConfigMap {
  private:
    class PrivateType {
      public:
        explicit PrivateType() = default;
    };

  public:
    GlobalConfigMap(PrivateType);

    static std::shared_ptr<GlobalConfigMap> get_singleton();

    bool empty() const noexcept;

    size_t size() const noexcept;

    size_t max_size() const noexcept;

    void clear() noexcept;

    size_t erase(std::string const &key);

    std::string  &at_string(std::string const &key);
    std::int64_t &at_int(std::string const &key);
    double       &at_double(std::string const &key);

    std::string const  &at_string(std::string const &key) const;
    std::int64_t const &at_int(std::string const &key) const;
    double const       &at_double(std::string const &key) const;

    std::string  &get_string(std::string const &key);
    std::int64_t &get_int(std::string const &key);
    double       &get_double(std::string const &key);

    std::string const  &get_string(std::string const &key) const;
    std::int64_t const &get_int(std::string const &key) const;
    double const       &get_double(std::string const &key) const;

    std::shared_ptr<ConfigMap<std::string>> get_string_map();
    std::shared_ptr<ConfigMap<std::int64_t>> get_int_map();
    std::shared_ptr<ConfigMap<double>> get_double_map();

    template<typename T>
    void attach(std::shared_ptr<T> &obs) {
        if constexpr (std::is_base_of_v<ConfigObserver<std::string>, T>) {
            str_map_->attach(obs);
        }

        if constexpr (std::is_base_of_v<ConfigObserver<std::int64_t>, T>) {
            int_map_->attach(obs);
        }

        if constexpr (std::is_base_of_v<ConfigObserver<double>, T>) {
            double_map_->attach(obs);
        }
    }

    template<typename T>
    void detach(std::shared_ptr<T> &obs) {
        if constexpr (std::is_base_of_v<ConfigObserver<std::string>, T>) {
            str_map_->detach(obs);
        }

        if constexpr (std::is_base_of_v<ConfigObserver<std::int64_t>, T>) {
            int_map_->detach(obs);
        }

        if constexpr (std::is_base_of_v<ConfigObserver<double>, T>) {
            double_map_->detach(obs);
        }
    }

    void notify();

  private:
    explicit GlobalConfigMap();

    std::shared_ptr<ConfigMap<std::string>>  str_map_;
    std::shared_ptr<ConfigMap<std::int64_t>> int_map_;
    std::shared_ptr<ConfigMap<double>>       double_map_;
};

} // namespace einsums

template <class Value>
bool operator==(std::unordered_map<std::string, Value> const &lhs, einsums::ConfigMap<Value> const &rhs) {
    return lhs == rhs.get_map();
}

template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, std::unordered_map<std::string, Value> const &rhs) {
    return lhs.get_map() == rhs;
}

template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, einsums::ConfigMap<Value> const &rhs) {
    return lhs.get_map() == rhs.get_map();
}
