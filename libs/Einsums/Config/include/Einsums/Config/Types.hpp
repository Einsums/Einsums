//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/DesignPatterns/Singleton.hpp>

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace einsums {

#ifndef DOXYGEN
// Forward declaration.
template <typename Value>
class ConfigObserver;
#endif

/**
 * @class ConfigMap
 *
 * @brief Holds a mapping of string keys to configuration values.
 *
 * Objects of this type can hold maps of configuration variables. They can also act as a subject,
 * which can attach observers. When a configuration variable is updated, this map will notify its
 * observers with the new information. it has all of the methods and typedefs available from std::map.
 *
 * @tparam Value The type of data to be associated with each key.
 */
template <typename Value>
class ConfigMap : public std::enable_shared_from_this<ConfigMap<Value>> {
  private:
    /**
     * @class PrivateType
     *
     * @brief This class allows for a public constructor that can't be used in public contexts.
     *
     * This class helps users to make shared pointers from this class.
     */
    class PrivateType {
      public:
        explicit PrivateType() = default;
    };

  public:
    /**
     * @typedef MappingType
     *
     * @brief Represents the type used to hold the option map.
     */
    using MappingType = std::unordered_map<std::string, Value>;

    /**
     * @defgroup mapdefs Map Definitions
     *
     * These are definitions that are made visible from the underlying mapping type.
     */
    ///@{
    /**
     * @typedef key_type
     *
     * @brief The type of the keys in the map. Should be strings.
     */
    using key_type = typename MappingType::key_type;

    /**
     * @typedef mapped_type
     *
     * @brief The type of the values stored in the map.
     */
    using mapped_type = typename MappingType::mapped_type;

    /**
     * @typedef value_type
     *
     * @brief The type holding key-value pairs.
     */
    using value_type = typename MappingType::value_type;

    /**
     * @typedef size_type
     *
     * @brief The return type of size query operations. Usually size_t.
     */
    using size_type = typename MappingType::size_type;

    /**
     * @typedef difference_type
     *
     * @brief The return type of some operations that need to represent sizes and negative values together. Usually ptrdiff_t.
     */
    using difference_type = typename MappingType::difference_type;

    /**
     * @typedef hasher
     *
     * @brief The type used to compute the has values of the keys.
     */
    using hasher = typename MappingType::hasher;

    /**
     * @typedef key_equal
     *
     * @brief The type used to compare keys to see if they are equal or not.
     */
    using key_equal = typename MappingType::key_equal;

    /**
     * @typedef allocator_type
     *
     * @brief The type used for allocating the space for the map.
     */
    using allocator_type = typename MappingType::allocator_type;

    /**
     * @typedef reference
     *
     * @brief The type used for modifiable references as return values.
     */
    using reference = typename MappingType::reference;

    /**
     * @typedef const_reference
     *
     * @brief The type used for non-modifiable references as return values.
     */
    using const_reference = typename MappingType::const_reference;

    /**
     * @typedef pointer
     *
     * @brief The type used for pointers as return values.
     */
    using pointer = typename MappingType::pointer;

    /**
     * @typedef const_pointer
     *
     * @brief The type used for constant pointers as return values.
     */
    using const_pointer = typename MappingType::const_pointer;

    /**
     * @typedef iterator
     *
     * @brief The type used for iterators over the map.
     */
    using iterator = typename MappingType::iterator;

    /**
     * @typedef const_iterator
     *
     * @brief The type used for iterators over the map that don't modify the underlying data.
     */
    using const_iterator = typename MappingType::const_iterator;

    /**
     * @typedef local_iterator
     *
     * @brief Iterator that iterates over elements in a bucket.
     */
    using local_iterator = typename MappingType::local_iterator;

    /**
     * @typedef const_local_iterator
     *
     * @brief Iterator that iterates over elements in a bucket without modifying the underlying data.
     */
    using const_local_iterator = typename MappingType::const_local_iterator;

    /**
     * @typedef node_type
     *
     * @brief The type used to store the data internally.
     */
    using node_type = typename MappingType::node_type;

    /**
     * @typedef insert_return_type
     *
     * @brief The type returned from an insert operation.
     */
    using insert_return_type = typename MappingType::insert_return_type;
    ///@}

    /**
     * @typedef Key
     *
     * @brief The type used to access elements within the map.
     */
    using Key = std::string;

    /**
     * Public constructor that can only be accessed in private contexts. Used to make shared pointers
     * from this class.
     */
    ConfigMap(PrivateType) : map_() {}

    /**
     * @brief Attach an observer to this subject.
     *
     * @param obs The observer to attach.
     */
    void attach(std::shared_ptr<ConfigObserver<Value>> obs) { observers_.emplace_back(obs); }

    /**
     * @brief Detach an observer from this subject.
     *
     * @param obs The observer to detach.
     */
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

    /**
     * @brief Go through the observers and update their state with the current state of the map.
     */
    void notify() {
        for (auto &obs : this->observers_) {
            obs.lock()->update(this->shared_from_this());
        }
    }

    /**
     * @brief Create a shared pointer from this class.
     *
     * @return A shared pointer to a ConfigMap.
     */
    static std::shared_ptr<ConfigMap<Value>> create() { return std::make_shared<ConfigMap<Value>>(PrivateType()); }

    // Map wrapping.
    /**
     * @addtogroup mapdefs
     */
    ///@{

    /**
     * @brief Gets the allocator used for this mapping.
     *
     * @return the allocator that is used to allocate space for this mapping.
     */
    inline allocator_type get_allocator() const noexcept { return map_.get_allocator(); }

    /**
     * @brief Get an iterator to the beginning of the map.
     *
     * @return An iterator pointing to the beginning of the map.
     */
    inline iterator begin() noexcept { return map_.begin(); }

    /**
     * @copydoc ConfigMap::begin()
     */
    inline const_iterator begin() const noexcept { return map_.begin(); }

    /**
     * @copydoc ConfigMap::begin()
     */
    inline const_iterator cbegin() const noexcept { return map_.cbegin(); }

    /**
     * @brief Gets an iterator pointing to the element after the final element in the map.
     *
     * @return The iterator representing the end of the map.
     */
    inline iterator end() noexcept { return map_.end(); }

    /**
     * @copydoc ConfigMap::end()
     */
    inline const_iterator end() const noexcept { return map_.end(); }

    /**
     * @copydoc ConfigMap::end()
     */
    inline const_iterator cend() const noexcept { return map_.cend(); }

    /**
     * @brief Check to see if the map is empty.
     *
     * @return True if the map is empty, false if there are elements in the map.
     */
    inline bool empty() const noexcept { return map_.empty(); }

    /**
     * @brief Get the number of elements in the map.
     *
     * @return The number of elements in the map.
     */
    inline size_type size() const noexcept { return map_.size(); }

    /**
     * @brief Get the maximum size the map is able to be.
     *
     * @return The maximum size the map can be.
     */
    inline size_type max_size() const noexcept { return map_.max_size(); }

    /**
     * @brief Remove all elements from the map.
     */
    inline void clear() noexcept { map_.clear(); }

    /**
     * @brief Insert the key-value pair into the map.
     *
     * @param value The key-value pair to insert.
     *
     * @return A pair containing an iterator pointing to the position the element was inserted into, as well as a flag
     * indicating whether the insertion was successful.
     */
    inline std::pair<iterator, bool> insert(value_type const &value) { return map_.insert(value); }

    /**
     * @copydoc ConfigMap::insert(value_type const&)
     */
    inline std::pair<iterator, bool> insert(value_type &&value) { return map_.insert(std::forward<value_type>(value)); }

    /**
     * @copydoc ConfigMap::insert(value_type const&)
     */
    template <class P>
    std::pair<iterator, bool> insert(P &&value) {
        return map_.insert(std::forward<P>(value));
    }

    /**
     * @brief Insert the key-value pair into the  map.
     *
     * Inserts a key-value pair into the map with a hint telling where the value may be able to go.
     *
     * @param hint A hint for where the pair may be able to be inserted.
     * @param value The key-value pair to insert.
     *
     * @return An iterator pointing to where the key-value pair was inserted.
     */
    inline iterator insert(const_iterator hint, value_type const &value) { return map_.insert(hint, value); }

    /**
     * @copydoc ConfigMap::insert(const_iterator,value_type const&)
     */
    inline iterator insert(const_iterator hint, value_type &&value) { return map_.insert(hint, std::forward<value_type>(value)); }

    /**
     * @copydoc ConfigMap::insert(const_iterator,value_type const&)
     */
    template <class P>
    iterator insert(const_iterator hint, P &&value) {
        return map_.insert(hint, std::forward<P>(value));
    }

    /**
     * @brief Insert several key-value pairs.
     *
     * @param first An iterator to the first item to insert.
     * @param last An iterator to the last item to insert.
     */
    template <class InputIt>
    void insert(InputIt first, InputIt last) {
        map_.insert(first, last);
    }

    /**
     * @brief Insert several key-value pairs
     *
     * @param ilist A list of key-value pairs to insert.
     */
    inline void insert(std::initializer_list<value_type> ilist) { map_.insert(ilist); }

    /**
     * @brief Insert a node into the map.
     *
     * @param nh The node handle to insert.
     *
     * @return A structured object containing the position of the node in the list, a flag telling
     * whether the item was inserted, and the node that was inserted.
     */
    inline insert_return_type insert(node_type &&nh) { return map_.insert(std::forward<node_type>(nh)); }

    /**
     * @brief Insert a node into the map with a hint.
     *
     * @param hint A hint telling where the node may be inserted.
     * @param nh The node handle to insert.
     *
     * @return An iterator to the inserted node.
     */
    inline iterator insert(const_iterator hint, node_type &&nh) { return map_.insert(hint, std::forward<node_type>(nh)); }

    /**
     * @brief Inserts or assigns a value to a key.
     *
     * If the map contains the key, then the value at the key will be assigned to the value passed.
     * If the map doesn't contain the key, then the key is added to the map and its value is set to the
     * value passed in.
     *
     * @param key The key to insert.
     * @param obj The value to set at the key.
     *
     * @return A pair containing an iterator to the position the key was inserted into, and a flag
     * telling whether the insertion took place. If the flag is false, no insertion took place, but
     * rather an assignment.
     */
    template <class M>
    std::pair<iterator, bool> insert_or_assign(Key const &key, M &&obj) {
        return map_.insert_or_assign(key, std::forward<M>(obj));
    }

    /**
     * @copydoc ConfigMap::insert_or_assign(Key const&,M&&)
     */
    template <class M>
    std::pair<iterator, bool> insert_or_assign(Key &&key, M &&obj) {
        return map_.insert_or_assign(std::forward<Key>(key), std::forward<M>(obj));
    }

    /**
     * @brief Inserts or assigns a value to a key.
     *
     * If the map contains the key, then the value at the key will be assigned to the value passed.
     * If the map doesn't contain the key, then the key is added to the map and its value is set to the
     * value passed in.
     *
     * @param hint An iterator telling where the key may be placed.
     * @param key The key to insert.
     * @param obj The value to set at the key.
     *
     * @return A pair containing an iterator to the position the key was inserted into, and a flag
     * telling whether the insertion took place. If the flag is false, no insertion took place, but
     * rather an assignment.
     */
    template <class M>
    iterator insert_or_assign(const_iterator hint, Key const &key, M &&obj) {
        return map_.insert_or_assign(hint, key, std::forward<M>(obj));
    }

    /**
     * @copydoc ConfigMap::insert_or_assign(const_iterator,Key const&,M&&)
     */
    template <class M>
    iterator insert_or_assign(const_iterator hint, Key &&key, M &&obj) {
        return map_.insert_or_assign(hint, std::forward<Key>(key), std::forward<M>(obj));
    }

    /**
     * @brief Construct a new object at the specified key. The key should be the first argument.
     *
     * @tparam Args The types of the arguments.
     *
     * @param args The arguments to pass to the constructor. The first argument is the key for the insertion.
     *
     * @return A pair containing an iterator pointing to the element that was constructed, as well as a flag
     * telling whether the item was inserted or not.
     */
    template <class... Args>
    std::pair<iterator, bool> emplace(Args &&...args) {
        return map_.emplace(std::forward<Args>(args)...);
    }

    /**
     * @brief Construct a new object at the specified key. The key should be the first argument.
     *
     * @param hint A hint to tell where the key may be put.
     * @param args The arguments to pass to the constructor. The first argument is the key for the insertion.
     *
     * @return A pair containing an iterator pointing to the element that was constructed, as well as a flag
     * telling whether the item was inserted or not.
     */
    template <class... Args>
    iterator emplace_hint(const_iterator hint, Args &&...args) {
        return map_.emplace_hint(hint, std::forward<Args>(args)...);
    }

    /**
     * @brief Construct a new object at the specified key. The key should be the first argument.
     *
     * Normally, emplace may try to construct an object, even if insertion would fail. This version does not
     * try to construct the object if the insertion would fail.
     *
     * @param key The key to use for the placement.
     * @param args The arguments to pass to the constructor.
     *
     * @return A pair containing an iterator pointing to the element that was constructed, as well as a flag
     * telling whether the item was inserted or not.
     */
    template <class... Args>
    std::pair<iterator, bool> try_emplace(Key const &key, Args &&...args) {
        return map_.try_emplace(key, std::forward<Args>(args)...);
    }

    /**
     * @brief Construct a new object at the specified key. The key should be the first argument.
     *
     * Normally, emplace may try to construct an object, even if insertion would fail. This version does not
     * try to construct the object if the insertion would fail.
     *
     * @param key The key to use for the placement.
     * @param args The arguments to pass to the constructor.
     *
     * @return A pair containing an iterator pointing to the element that was constructed, as well as a flag
     * telling whether the item was inserted or not.
     */
    template <class... Args>
    std::pair<iterator, bool> try_emplace(Key &&key, Args &&...args) {
        return map_.try_emplace(std::forward<Key>(key), std::forward<Args>(args)...);
    }

    /**
     * @brief Construct a new object at the specified key. The key should be the first argument.
     *
     * Normally, emplace may try to construct an object, even if insertion would fail. This version does not
     * try to construct the object if the insertion would fail.
     *
     * @param hint A hint to tell where the key may be put.
     * @param key The key to use for the emplacement.
     * @param args The arguments to pass to the constructor. The first argument is the key for the insertion.
     *
     * @return A pair containing an iterator pointing to the element that was constructed, as well as a flag
     * telling whether the item was inserted or not.
     */
    template <class... Args>
    iterator try_emplace(const_iterator hint, Key const &key, Args &&...args) {
        return map_.try_emplace(hint, key, std::forward<Args>(args)...);
    }

    /**
     * @brief Construct a new object at the specified key. The key should be the first argument.
     *
     * Normally, emplace may try to construct an object, even if insertion would fail. This version does not
     * try to construct the object if the insertion would fail.
     *
     * @param hint A hint to tell where the key may be put.
     * @param key The key to use for the emplacement.
     * @param args The arguments to pass to the constructor. The first argument is the key for the insertion.
     *
     * @return A pair containing an iterator pointing to the element that was constructed, as well as a flag
     * telling whether the item was inserted or not.
     */
    template <class... Args>
    iterator try_emplace(const_iterator hint, Key &&key, Args &&...args) {
        return map_.try_emplace(hint, std::forward<Key>(key), std::forward<Args>(args)...);
    }

    /**
     * @brief Erase the value at the given position.
     *
     * @param pos An iterator pointing to the element to erase.
     *
     * @return An iterator to the element after the removed element.
     */
    inline iterator erase(iterator pos) { return map_.erase(pos); }

    /**
     * @copydoc ConfigMap::erase(iterator)
     */
    inline iterator erase(const_iterator pos) { return map_.erase(pos); }

    /**
     * @brief Erase several elements from the map.
     *
     * @param first An iterator to the first element to erase.
     * @param last An iterator to the end of the list of elements to erase.
     *
     * @return An iterator to the element after the last element removed.
     */
    inline iterator erase(const_iterator first, const_iterator last) { return map_.erase(first, last); }

    /**
     * @brief Erase the entry for a key.
     *
     * @param key The key to erase.
     *
     * @return The number of elements removed.
     */
    inline size_type erase(Key const &key) { return map_.erase(key); }

    /**
     * @brief Swap the data between maps.
     *
     * @param other The map to swap with.
     */
    inline void swap(MappingType &other) noexcept(noexcept(std::allocator_traits<allocator_type>::is_always_equal::value &&
                                                           std::is_nothrow_swappable<hasher>::value &&
                                                           std::is_nothrow_swappable<key_equal>::value)) {
        map_.swap(other);
    }

    /**
     * @copydoc ConfigMap::swap(MappingType &)
     */
    inline void swap(ConfigMap<Value> &other) noexcept(noexcept(std::allocator_traits<allocator_type>::is_always_equal::value &&
                                                                std::is_nothrow_swappable<hasher>::value &&
                                                                std::is_nothrow_swappable<key_equal>::value)) {
        map_.swap(other.map_);
    }

    /**
     * @brief Grab the node handle at the specified position.
     *
     * @param pos An iterator pointing to the position to extract.
     *
     * @return The node handle for the specified position.
     */
    inline node_type extract(const_iterator pos) { return map_.extract(pos); }

    /**
     * @brief Grab the node handle for the specified key.
     *
     * @param key The key for the extraction.
     *
     * @return The node handle for the given key.
     */
    inline node_type extract(Key const &key) { return map_.extract(key); }

    /**
     * @brief Merge the data from another map into this map.
     *
     * @param other The map with data to copy and merge.
     */
    template <class H2, class P2>
    void merge(std::unordered_map<Key, Value, H2, P2, allocator_type> &other) {
        map_.merge(other);
    }

    /**
     * @brief Merge the data from another map into this map.
     *
     * @param other The map with data to move and merge.
     */
    template <class H2, class P2>
    void merge(std::unordered_map<Key, Value, H2, P2, allocator_type> &&other) {
        map_.merge(std::forward<std::unordered_map<Key, Value, H2, P2, allocator_type>>(other));
    }

    /**
     * @brief Merge the data from another map into this map.
     *
     * @param other The map with data to copy and merge.
     */
    template <class H2, class P2>
    void merge(std::unordered_multimap<Key, Value, H2, P2, allocator_type> &other) {
        map_.merge(other);
    }

    /**
     * @brief Merge the data from another map into this map.
     *
     * @param other The map with data to move and merge.
     */
    template <class H2, class P2>
    void merge(std::unordered_multimap<Key, Value, H2, P2, allocator_type> &&other) {
        map_.merge(std::forward<std::unordered_multimap<Key, Value, H2, P2, allocator_type>>(other));
    }

    /**
     * @brief Merge the data from another map into this map.
     *
     * @param other The map with data to copy and merge.
     */
    inline void merge(ConfigMap<Value> &other) { map_.merge(other.map_); }

    /**
     * @brief Merge the data from another map into this map.
     *
     * @param other The map with data to move and merge.
     */
    void merge(ConfigMap<Value> &&other) { map_.merge(std::forward<typename ConfigMap<Value>::MappingType>(other._map)); }

    /**
     * @brief Gets the value at the specified key. Throws an error if no value exists.
     *
     * @param key The key to query.
     *
     * @return The value corresponding to the given key.
     *
     * @throws std::out_of_range Throws when the key is not in the map.
     */
    inline Value &at(Key const &key) { return map_.at(key); }

    /**
     * @copydoc ConfigMap::at()
     */
    inline Value const &at(Key const &key) const { return map_.at(key); }

    /**
     * @brief Gets the value at the specified key. Inserts an element if no value exists.
     *
     * @param key The key to query.
     *
     * @return The value corresponding to the given key.
     */
    inline Value &operator[](Key const &key) { return map_[key]; }

    /**
     * @copydoc ConfigMap::operator[]()
     */
    inline Value &operator[](Key &&key) const { return map_[key]; }

    /**
     * @brief Gets the number of elements corresponding to the given key. Can be zero or one.
     *
     * @param key The key to query.
     *
     * @return 0 if the key is not in the map, 1 if it is in the map.
     */
    inline size_type count(Key const &key) const { return map_.count(key); }

    /**
     * @copydoc ConfigMap::count()
     */
    template <class K>
    size_type count(K const &key) const {
        return map_.count(key);
    }

    /**
     * @brief Finds the position of the given key in the map.
     *
     * @param key The key to find.
     *
     * @return An iterator pointing to the key, or ConfigMap::end() if the key is not in the map.
     */
    inline iterator find(Key const &key) { return map_.find(key); }

    /**
     * @copydoc ConfigMap::find()
     */
    inline const_iterator find(Key const &key) const { return map_.find(key); }

    /**
     * @copydoc ConfigMap::find()
     */
    template <class K>
    iterator find(K const &key) {
        return map_.find(key);
    }

    /**
     * @copydoc ConfigMap::find()
     */
    template <class K>
    const_iterator find(K const &key) const {
        return map_.find(key);
    }

    /**
     * @brief Tests whether the key is in the map.
     *
     * @param key The key to find.
     *
     * @return True if the key is in the map. False otherwise.
     */
    inline bool contains(Key const &key) const { return map_.contains(key); }

    /**
     * @copydoc ConfigMap::contains()
     */
    template <class K>
    bool contains(K const &key) const {
        return map_.contains(key);
    }

    /**
     * @brief Returns the range of elements that match the given key.
     *
     * @param key The key to query.
     *
     * @return A pair containing iterators pointing to the start and end of the range.
     */
    inline std::pair<iterator, iterator> equal_range(Key const &key) { return map_.equal_range(key); }

    /**
     * @copydoc ConfigMap::equal_range()
     */
    inline std::pair<const_iterator, const_iterator> equal_range(Key const &key) const { return map_.equal_range(key); }

    /**
     * @copydoc ConfigMap::equal_range()
     */
    template <class K>
    std::pair<iterator, iterator> equal_range(K const &key) {
        return map_.equal_range(key);
    }

    /**
     * @copydoc ConfigMap::equal_range()
     */
    template <class K>
    std::pair<const_iterator, const_iterator> equal_range(K const &key) const {
        return map_.equal_range(key);
    }

    /**
     * @brief Gets an iterator to the start of the specified bucket.
     *
     * @param n The bucket to query.
     *
     * @return The iterator to the start of the given bucket.
     */
    inline local_iterator begin(size_type n) { return map_.begin(n); }

    /**
     * @copydoc ConfigMap::begin(size_type)
     */
    inline const_local_iterator begin(size_type n) const { return map_.begin(n); }

    /**
     * @copydoc ConfigMap::begin(size_type)
     */
    inline const_local_iterator cbegin(size_type n) const { return map_.cbegin(n); }

    /**
     * @brief Gets an iterator to the end of the given bucket.
     *
     * @param n The bucket to query.
     *
     * @return The iterator to the end of the given bucket.
     */
    inline local_iterator end(size_type n) { return map_.end(n); }

    /**
     * @copydoc ConfigMap::end(size_type)
     */
    inline const_local_iterator end(size_type n) const { return map_.end(n); }

    /**
     * @copydoc ConfigMap::end(size_type)
     */
    inline const_local_iterator cend(size_type n) const { return map_.cend(n); }

    /**
     * @brief Get the number of buckets.
     *
     * @return The number of buckets.
     */
    inline size_type bucket_count() const { return map_.bucket_count(); }

    /**
     * @brief Get the maximum number of buckets.
     *
     * @return The maximum number of buckets.
     */
    inline size_type max_bucket_count() const { return map_.max_bucket_count(); }

    /**
     * @brief Get the size of the given bucket.
     *
     * @param n The bucket to query.
     *
     * @return The seize of the requested bucket.
     */
    inline size_type bucket_size(size_type n) const { return map_.bucket_size(n); }

    /**
     * @brief Get the bucket that contains a given key.
     *
     * @param key The key to search for.
     *
     * @return The bucket that contains the key.
     */
    inline size_type bucket(Key const &key) const { return map_.bucket(key); }

    /**
     * @brief Get the average number of elements per bucket.
     *
     * @return The load factor for the map, or the average number of elements per bucket.
     */
    inline float load_factor() const { return map_.load_factor(); }

    /**
     * @brief Get the maximum allowed load factor.
     *
     * @return The maximum allowed load factor.
     */
    inline float max_load_factor() const { return map_.max_load_factor(); }

    /**
     * @brief Sets the maximum allowed load factor.
     *
     * @param ml The new load factor.
     */
    inline void max_load_factor(float ml) { map_.max_load_factor(ml); }

    /**
     * @brief Reserves space for the given number of buckets, then recalculates the hash table.
     *
     * @param count The number of buckets to reserve space for.
     */
    inline void rehash(size_type count) { map_.rehash(count); }

    /**
     * @brief Reserves space for the given number of elements, then recalculates the hash table.
     *
     * @param count The number of elements to reserve space for.
     */
    inline void reserve(size_type count) { map_.reserve(count); }

    /**
     * @brief Get the function used to calculate the hash values.
     *
     * @return The function used to calculate the hash values.
     */
    inline hasher hash_function() const { return map_.hash_function(); }

    /**
     * @brief Get the function used to check whether two keys are equal.
     *
     * @return The function used to check whether two keys are equal.
     */
    inline key_equal key_eq() const { return map_.key_eq(); }
    ///@}

    // Other methods.

    /**
     * @brief Get the underlying unordered map that stores the data.
     *
     * @return The underlying unordered map.
     */
    MappingType &get_map() { return map_; }

    /**
     * @copydoc ConfigMap::get_map()
     */
    MappingType const &get_map() const { return map_; }

    /**
     * @brief Convert to an unordered map reference.
     */
    operator MappingType &() { return map_; }

    /**
     * @brief Convert to an unordered map const reference.
     */
    operator MappingType const &() const { return map_; }

  private:
    /**
     * @brief Default constructor.
     */
    explicit ConfigMap() = default;

    /**
     * @property map_
     *
     * @brief The map that stores all of the keys and values.
     */
    std::unordered_map<std::string, Value> map_;

    /** 
     * @property observers_
     *
     * @brief The list of weak pointers to all of the observers of this map.
     */
    std::list<std::weak_ptr<ConfigObserver<Value>>> observers_;

    friend class std::shared_ptr<ConfigObserver<Value>>;
};

/**
 * @typedef SharedConfigMap
 *
 * @brief Shared pointer to a ConfigMap.
 */
template <typename Value>
using SharedConfigMap = std::shared_ptr<ConfigMap<Value>>;
// using SharedInfoMap = std::shared_ptr<InfoMap>;

/**
 * @class ConfigObserver
 *
 * @brief Represents an object that can observe a ConfigMap.
 *
 * Whenever the mapping the object is observing updates, the observer will receive a
 * notification to update its state as well.
 */
template <typename Value>
class ConfigObserver {
  public:
    /**
     * @brief Default destructor.
     */
    virtual ~ConfigObserver() = default;

    /**
     * @brief Update the state of the observer with the given config map.
     *
     * @param map The map to use to update the state of the observer.
     */
    virtual void update(SharedConfigMap<Value> map) = 0;
};

/**
 * @class GlobalConfigMap
 *
 * @brief This is a map that holds global configuration variables.
 *
 * This map holds three ConfigMap's inside. It has one for each of integer values, floating point values,
 * and string values. Observers can observe this map, and depending on the type of the observer, it will
 * be attached to the appropriate sub-map. This class is a singleton.
 */
class EINSUMS_EXPORT GlobalConfigMap {
    EINSUMS_SINGLETON_DEF(GlobalConfigMap)
  public:
    /**
     * @brief Checks to see if the map is empty.
     */
    bool empty() const noexcept;

    /**
     * @brief Gets the size of the map.
     */
    size_t size() const noexcept;

    /**
     * @brief Gets the maximum number of buckets in the map.
     */
    size_t max_size() const noexcept;

    /**
     * @brief Clears the map.
     */
    void clear() noexcept;

    /**
     * @brief Removes the entry at a given key.
     *
     * @param key The key to remove.
     */
    size_t erase(std::string const &key);

    /**
     * @brief Get the string value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::string  &at_string(std::string const &key);

    /**
     * @brief Get the integer value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::int64_t &at_int(std::string const &key);

    /**
     * @brief Get the floating point value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    double       &at_double(std::string const &key);

    /**
     * @brief Get the string value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::string const  &at_string(std::string const &key) const;

    /**
     * @brief Get the integer value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::int64_t const &at_int(std::string const &key) const;

    /**
     * @brief Get the floating point value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    double const       &at_double(std::string const &key) const;

    /**
     * @brief Get the string value stored at the given key.
     *
     * Adds the key to the map if it doesn't exist.
     *
     * @param key The key to query.
     */
    std::string  &get_string(std::string const &key);

    /**
     * @brief Get the integer value stored at the given key.
     *
     * Adds the key to the map if it doesn't exist.
     *
     * @param key The key to query.
     */
    std::int64_t &get_int(std::string const &key);

    /**
     * @brief Get the floating point value stored at the given key.
     *
     * Adds the key to the map if it doesn't exist.
     *
     * @param key The key to query.
     */
    double       &get_double(std::string const &key);

    /**
     * @brief Get the string value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::string const  &get_string(std::string const &key) const;

    /**
     * @brief Get the integer value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::int64_t const &get_int(std::string const &key) const;

    /**
     * @brief Get the floating point value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    double const       &get_double(std::string const &key) const;

    /**
     * @brief Returns the map containing string options.
     */
    std::shared_ptr<ConfigMap<std::string>>  get_string_map();

    /**
     * @brief Returns the map containing integer options.
     */
    std::shared_ptr<ConfigMap<std::int64_t>> get_int_map();

    /**
     * @brief Returns the map containing floating point options.
     */
    std::shared_ptr<ConfigMap<double>>       get_double_map();

    /**
     * @brief Attach an observer to the global configuration map.
     *
     * The observer should be an object derived from ConfigObserver. The template parameter
     * on the ConfigObserver class
     * determines which map or maps the observer will be attached to. The template parameter can
     * be either @c std::string , @c std::int64_t , or @c double . If the observer derives from
     * multiple of these observers, it will be attached to each map that it is able to.
     *
     * @param obs The observer to attach.
     */
    template <typename T>
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

    /**
     * @brief Detach an observer from the global configuration map.
     *
     * @param obs The observer to remove.
     */
    template <typename T>
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

    /**
     * @brief Update all of the observers of the map that something has changed.
     */
    void notify();

  private:
    explicit GlobalConfigMap();

    /**
     * @property str_map_
     *
     * @brief Holds the string valued options.
     */
    std::shared_ptr<ConfigMap<std::string>>  str_map_;

    /**
     * @property int_map_
     *
     * @brief Holds the integer valued options.
     */
    std::shared_ptr<ConfigMap<std::int64_t>> int_map_;

    /**
     * @property double_map_
     *
     * @brief Holds the floating-point valued options.
     */
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
