//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Iterator/Enumerate.hpp>

#include <iterator>
#include <memory>
#include <numbers>
#include <numeric>
#include <utility>

namespace einsums {

/**
 * @struct ZipIter
 *
 * @brief An iterator that iterates over several containers simultaneously.
 */
template <typename... ContainerIters>
struct ZipIter final {
  public:
    /**
     * @typedef difference_type
     *
     * @brief The type indicating the distance between two iterators.
     */
    using difference_type = ptrdiff_t;

    /**
     * @typedef value_type
     *
     * @brief The values returned by the iterator.
     */
    using value_type = std::tuple<typename std::iterator_traits<ContainerIters>::reference...>;

    /**
     * @typedef reference
     *
     * @brief Modifiable version of the return value of the iterator.
     */
    using reference = std::tuple<typename std::iterator_traits<ContainerIters>::reference...>;

    /**
     * @typedef iterator_category
     *
     * @brief Indicates that this is a forward iterator.
     */
    using iterator_category = std::forward_iterator_tag;

    /**
     * @brief Default constructor.
     */
    constexpr ZipIter() : current_{} {}

    /**
     * @brief Copy constructor.
     */
    constexpr ZipIter(ZipIter const &other) : current_{other.current_} {}

    /**
     * @brief Copy assignment.
     */
    constexpr ZipIter &operator=(ZipIter const &other) {
        current_ = other.current_;
        return *this;
    }

    /**
     * @brief Initialize the iterator with the given iterators.
     */
    constexpr ZipIter(std::tuple<ContainerIters &...> const &containers) {
        init_impl(containers, std::make_index_sequence<sizeof...(ContainerIters)>());
    }

    /**
     * @brief Initialize the iterator with the given iterators.
     */
    constexpr ZipIter(ContainerIters const &...containers) : current_{containers...} {}

    /**
     * @brief Dereference the iterator.
     */
    constexpr value_type operator*() const { return deref_impl(std::make_index_sequence<sizeof...(ContainerIters)>()); }

    /**
     * @brief Dereference the iterator.
     */
    constexpr reference operator*() { return deref_impl_ref(std::make_index_sequence<sizeof...(ContainerIters)>()); }

    /**
     * @brief Increment the iterator.
     */
    constexpr ZipIter &operator++() {
        for_sequence<sizeof...(ContainerIters)>([this](auto n) { std::get<(size_t)n>(current_)++; });
        return *this;
    }

    /**
     * @brief Increment the iterator.
     */
    constexpr ZipIter &operator++(int) {
        for_sequence<sizeof...(ContainerIters)>([this](auto n) { std::get<(size_t)n>(current_)++; });
        return *this;
    }

    /**
     * @brief Check whether any of the iterators are equal.
     *
     * This uses an or mask to make sure that we don't go past any of the ends of any of the
     * iterators. If we used an and mask and we compared with the iterator containing the ends
     * of the containers, then comparison will fail if any iterator is not finished.
     */
    constexpr bool operator==(ZipIter const &other) const {
        return equal_impl(other, std::make_index_sequence<sizeof...(ContainerIters)>());
    }

    /**
     * @brief Check whether all of the iterators are not equal.
     *
     * This uses an and mask to make sure that we don't go past any of the ends of any of the
     * iterators. If we used an or mask and we compared with the iterator containing the ends
     * of the containers, then comparison will fail if any iterator is not finished.
     */
    constexpr bool operator!=(ZipIter const &other) const {
        return !equal_impl(other, std::make_index_sequence<sizeof...(ContainerIters)>());
    }

    /**
     * @brief Swap the contents of two iterators.
     */
    constexpr void swap(ZipIter &other) { std::swap(current_, other.current_); }

  private:
    template <size_t... I>
    constexpr void init_impl(std::tuple<ContainerIters &...> const &containers, std::index_sequence<I...> const &) {
        current_ = std::make_tuple(std::get<I>(containers).begin()...);
    }

    template <size_t... I>
    constexpr value_type deref_impl(std::index_sequence<I...> const &) const {
        return value_type(*std::get<I>(current_)...);
    }

    template <size_t... I>
    constexpr reference deref_impl_ref(std::index_sequence<I...> const &) {
        return value_type(*std::get<I>(current_)...);
    }

    template <size_t... I>
    constexpr bool equal_impl(ZipIter const &other, std::index_sequence<I...> const &) const {
        return ((std::get<I>(current_) == std::get<I>(other.current_)) || ... || false);
    }

    std::tuple<ContainerIters...> current_;
};

/**
 * @struct Zip
 *
 * @brief Works like Python's @c zip() function.
 */
template <typename... Containers>
struct Zip final {
  public:
    /**
     * @typedef value_type
     *
     * @brief The type of data this zip holds.
     */
    using value_type = std::tuple<typename Containers::reference...>;

    /**
     * @typedef reference
     *
     * @brief A reference to the value type.
     */
    using reference = std::tuple<typename Containers::reference...> &;

    /**
     * @typedef const_reference
     *
     * @brief A const reference to the value type.
     */
    using const_reference = std::tuple<typename Containers::const_reference...> const &;

    /**
     * @typedef iterator
     *
     * @brief The iterator type.
     */
    using iterator = ZipIter<decltype(std::declval<Containers>().begin())...>;

    /**
     * @typedef const_iterator
     *
     * @brief The const iterator type.
     */
    using const_iterator = ZipIter<typename Containers::const_iterator...> const;

    /**
     * @typedef difference_type
     *
     * @brief The type for distances between objects.
     */
    using difference_type = ptrdiff_t;

    /**
     * @typedef size_type
     *
     * @brief The type used for container sizes.
     */
    using size_type = size_t;

    /**
     * @brief Default constructor.
     */
    constexpr Zip() = default;

    /**
     * @brief Copy constructor.
     */
    constexpr Zip(Zip const &other) : containers_{other.containers_} {};

    /**
     * @brief Initialize the zip object.
     */
    constexpr Zip(Containers &...containers) : containers_(std::tie(containers...)){};

    /**
     * @brief Copy assignment.
     */
    constexpr Zip &operator=(Zip const &other) {
        containers_ = other.containers_;
        return *this;
    }

    /**
     * @brief The iterator pointing to the beginning of the container.
     */
    constexpr iterator begin() { return begin_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    /**
     * @brief The iterator pointing to the end of the container.
     */
    constexpr iterator end() { return end_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    /**
     * @brief The iterator pointing to the beginning of the container.
     */
    constexpr const_iterator begin() const { return cbegin_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    /**
     * @brief The iterator pointing to the end of the container.
     */
    constexpr const_iterator end() const { return cend_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    /**
     * @brief The const iterator pointing to the beginning of the container.
     */
    constexpr const_iterator cbegin() const { return cbegin_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    /**
     * @brief The const iterator pointing to the end of the container.
     */
    constexpr const_iterator cend() const { return cend_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    /**
     * @brief Compare two zip objects.
     */
    constexpr bool operator==(Zip<Containers...> const &other) const {
        return equal_impl(other, std::make_index_sequence<sizeof...(Containers)>());
    }

    /**
     * @brief Compare two zip objects.
     */
    constexpr bool operator!=(Zip<Containers...> const &other) const {
        return !equal_impl(other, std::make_index_sequence<sizeof...(Containers)>());
    }

    /**
     * @brief Swap the contents of two zip objects.
     */
    constexpr void swap(Zip &other) { std::swap(containers_, other.containers_); }

    /**
     * @brief Get the size of the zip object.
     */
    constexpr size_t size() const {
        if constexpr (sizeof...(Containers) == 0) {
            return 0;
        } else {
            size_t out = std::numeric_limits<size_t>::max();

            for_sequence<sizeof...(Containers)>([this, &out](auto n) {
                size_t next_size = std::get<(size_t)n>(containers_).size();

                if (next_size < out) {
                    out = next_size;
                }
            });

            return out;
        }
    }

    /**
     * @brief Get the maximum possible size for the zip object.
     */
    constexpr size_t max_size() const {
        if constexpr (sizeof...(Containers) == 0) {
            return 0;
        } else {
            size_t out = std::numeric_limits<size_t>::max();

            for_sequence<sizeof...(Containers)>([this, &out](auto n) {
                size_t next_size = std::get<(size_t)n>(containers_).max_size();

                if (next_size < out) {
                    out = next_size;
                }
            });

            return out;
        }
    }

    /**
     * @brief Check whether the zip object contains data.
     */
    constexpr bool empty() const { return begin() == end(); }

  private:
    template <size_t... I>
    constexpr iterator begin_impl(std::index_sequence<I...> const &) {
        return ZipIter(std::get<I>(containers_).begin()...);
    }

    template <size_t... I>
    constexpr iterator end_impl(std::index_sequence<I...> const &) {
        return ZipIter(std::get<I>(containers_).end()...);
    }

    template <size_t... I>
    constexpr const_iterator cbegin_impl(std::index_sequence<I...> const &) const {
        return ZipIter(std::get<I>(containers_).cbegin()...);
    }

    template <size_t... I>
    constexpr const_iterator cend_impl(std::index_sequence<I...> const &) const {
        return ZipIter(std::get<I>(containers_).cend()...);
    }

    template <size_t... I>
    constexpr bool equal_impl(Zip const &other, std::index_sequence<I...> const &) const {
        return ((std::get<I>(containers_) == std::get<I>(other.containers_)) && ... && true);
    }

    std::tuple<Containers &...> containers_;
};

#ifdef __cpp_deduction_guides
template <typename... Containers>
Zip(Containers &...) -> Zip<Containers...>;
#endif

} // namespace einsums

/**
 * @brief Overload for swap.
 */
namespace std {
template <typename... Containers>
constexpr void swap(einsums::ZipIter<Containers...> &first, einsums::ZipIter<Containers...> &second) {
    first.swap(second);
}

/**
 * @brief Overload for swap.
 */
template <typename... Containers>
constexpr void swap(einsums::Zip<Containers...> &first, einsums::Zip<Containers...> &second) {
    first.swap(second);
}
} // namespace std