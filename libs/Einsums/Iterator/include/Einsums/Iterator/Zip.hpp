//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iterator>
#include <memory>
#include <numbers>
#include <numeric>
#include <utility>

namespace einsums {

template <typename... ContainerIters>
struct ZipIter final {
  public:
    using difference_type   = ptrdiff_t;
    using value_type        = std::tuple<typename std::iterator_traits<ContainerIters>::reference...>;
    using reference         = std::tuple<typename std::iterator_traits<ContainerIters>::reference...>;
    using iterator_category = std::forward_iterator_tag;

    constexpr ZipIter() : current_{} {}
    constexpr ZipIter(ZipIter const &other) : current_{other.current_} {}

    constexpr ZipIter &operator=(ZipIter const &other) {
        current_ = other.current_;
        return *this;
    }

    constexpr ZipIter(std::tuple<ContainerIters &...> const &containers) {
        init_impl(containers, std::make_index_sequence<sizeof...(ContainerIters)>());
    }

    constexpr ZipIter(ContainerIters const &...containers) : current_{containers...} {}

    constexpr value_type operator*() const { return deref_impl(std::make_index_sequence<sizeof...(ContainerIters)>()); }

    constexpr reference operator*() { return deref_impl_ref(std::make_index_sequence<sizeof...(ContainerIters)>()); }

    constexpr ZipIter &operator++() {
        for_sequence<sizeof...(ContainerIters)>([this](auto n) { std::get<(size_t)n>(current_)++; });
        return *this;
    }

    constexpr ZipIter &operator++(int) {
        for_sequence<sizeof...(ContainerIters)>([this](auto n) { std::get<(size_t)n>(current_)++; });
        return *this;
    }

    constexpr bool operator==(ZipIter const &other) const {
        return equal_impl(other, std::make_index_sequence<sizeof...(ContainerIters)>());
    }

    constexpr bool operator!=(ZipIter const &other) const {
        return !equal_impl(other, std::make_index_sequence<sizeof...(ContainerIters)>());
    }

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

template <typename... Containers>
struct Zip final {
  public:
    using value_type      = std::tuple<typename Containers::reference...>;
    using reference       = std::tuple<typename Containers::reference...> &;
    using const_reference = std::tuple<typename Containers::const_reference...> const &;
    using iterator        = ZipIter<decltype(std::declval<Containers>().begin())...>;
    using const_iterator  = ZipIter<typename Containers::const_iterator...> const;
    using difference_type = ptrdiff_t;
    using size_type       = size_t;

    constexpr Zip() = default;

    constexpr Zip(Zip const &other) : containers_{other.containers_} {};

    constexpr Zip(Containers &...containers) : containers_(std::tie(containers...)) {};

    constexpr Zip &operator=(Zip const &other) {
        containers_ = other.containers_;
        return *this;
    }

    constexpr iterator begin() { return begin_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    constexpr iterator end() { return end_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    constexpr iterator begin() const { return cbegin_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    constexpr iterator end() const { return cend_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    constexpr iterator cbegin() const { return cbegin_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    constexpr iterator cend() const { return cend_impl(std::make_index_sequence<sizeof...(Containers)>()); }

    constexpr bool operator==(Zip<Containers...> const &other) const {
        return equal_impl(other, std::make_index_sequence<sizeof...(Containers)>());
    }

    constexpr bool operator!=(Zip<Containers...> const &other) const {
        return !equal_impl(other, std::make_index_sequence<sizeof...(Containers)>());
    }

    constexpr void swap(Zip &other) { std::swap(containers_, other.containers_); }

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

namespace std {
template <typename... Containers>
constexpr void swap(einsums::ZipIter<Containers...> &first, einsums::ZipIter<Containers...> &second) {
    first.swap(second);
}

template <typename... Containers>
constexpr void swap(einsums::Zip<Containers...> &first, einsums::Zip<Containers...> &second) {
    first.swap(second);
}
} // namespace std