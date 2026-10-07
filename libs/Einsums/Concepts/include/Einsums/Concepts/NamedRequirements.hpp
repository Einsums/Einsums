//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iterator>
namespace einsums {

/**
 * @concept Container
 *
 * Checks that a type satisfies the Container requirement.
 */
template <typename T>
concept Container = requires(T a, T b, T const ca, T const cb, T &ra, T &rb, T const &rca, T const &rcb, T &&rra, T &&rrb) {
    // Check that the types exist.
    typename T::value_type;
    typename T::reference;
    typename T::const_reference;
    typename T::iterator;
    typename T::const_iterator;
    typename T::difference_type;
    typename T::size_type;

    // Check the properties of the types.
    requires std::forward_iterator<typename T::iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::iterator>, typename T::value_type>;
    requires std::is_convertible_v<typename T::iterator, typename T::const_iterator>;
    requires std::forward_iterator<typename T::const_iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::const_iterator>, typename T::value_type>;
    requires std::is_integral_v<typename T::difference_type>;
    requires std::is_signed_v<typename T::difference_type>;
    requires std::is_integral_v<typename T::size_type>;
    requires std::is_unsigned_v<typename T::size_type>;
    requires sizeof(typename T::size_type) >= sizeof(typename T::difference_type);

    // Expressions.
    T();
    T(a);
    T(ca);
    a  = b;
    a  = ca;
    ra = a;
    ra = ca;
    a.~T();
    { a.begin() } -> std::same_as<typename T::iterator>;
    { ca.begin() } -> std::same_as<typename T::const_iterator>;
    { a.end() } -> std::same_as<typename T::iterator>;
    { ca.end() } -> std::same_as<typename T::const_iterator>;
    { a.cbegin() } -> std::same_as<typename T::const_iterator>;
    { a.cend() } -> std::same_as<typename T::const_iterator>;

    { a == b } -> std::same_as<bool>;
    { a == cb } -> std::same_as<bool>;
    { ca == b } -> std::same_as<bool>;
    { ca == cb } -> std::same_as<bool>;
    { a != b } -> std::same_as<bool>;
    { a != cb } -> std::same_as<bool>;
    { ca != b } -> std::same_as<bool>;
    { ca != cb } -> std::same_as<bool>;

    a.swap(b);
    ra.swap(b);
    a.swap(rb);
    ra.swap(rb);
    requires std::swappable<T>;

    { a.size() } -> std::same_as<typename T::size_type>;
    { ca.size() } -> std::same_as<typename T::size_type>;
    { a.max_size() } -> std::same_as<typename T::size_type>;
    { ca.max_size() } -> std::same_as<typename T::size_type>;

    { a.empty() } -> std::same_as<bool>;
    { ca.empty() } -> std::same_as<bool>;
};

namespace detail {

template <typename T>
constexpr inline bool HasAllocator = requires { typename T::allocator_type; };

template <typename T>
concept HasAllocatorConcept = HasAllocator<T>;

template <typename T>
struct AllocatorOf {
    using type = void;
};

template <Container T>
struct AllocatorOf<T> {
    using type = std::allocator<typename T::value_type>;
};

template <Container T>
    requires(HasAllocatorConcept<T>)
struct AllocatorOf<T> {
    using type = typename T::allocator_type;
};

template <typename Cont, typename Allocator = AllocatorOf<Cont>::type, typename SizeType = typename Cont::size_type,
          typename Iterator = typename Cont::iterator, typename ConstIterator = typename Cont::const_iterator,
          typename Ref = typename Cont::reference, typename ConstRef = typename Cont::const_ref, typename T = typename Cont::value_type>
concept SequenceContainerWork =
    requires(Cont v, Cont const cv, Iterator i, Iterator j, std::initializer_list<T> il, SizeType n, ConstIterator p, ConstIterator q,
             ConstIterator q1, ConstIterator q2, T &t1, T const &&t2, T &&rv) {
        // Constructors.
        Cont(n, t1);
        Cont(n, t2);
        Cont(i, j);
        Cont(il);

        // Expressions.
        { v = il } -> std::same_as<Cont &>;
        { v.emplace(p) } -> std::same_as<Iterator>;
        { v.insert(p, t1) } -> std::same_as<Iterator>;
        { v.insert(p, t2) } -> std::same_as<Iterator>;
        { v.insert(p, rv) } -> std::same_as<Iterator>;
        { v.insert(p, n, t1) } -> std::same_as<Iterator>;
        { v.insert(p, n, t2) } -> std::same_as<Iterator>;
        { v.insert(p, i, j) } -> std::same_as<Iterator>;
        { v.insert(p, il) } -> std::same_as<Iterator>;
        { v.erase(q) } -> std::same_as<Iterator>;
        { v.erase(q1, q2) } -> std::same_as<Iterator>;
        v.clear();
        v.assign(i, j);
        v.assign(il);
        v.assign(n, t1);
        v.assign(n, t2);
    };

} // namespace detail

template <typename Cont>
concept SequenceContainer = Container<Cont> && detail::SequenceContainerWork<Cont>;

// This one isn't a true requirement, but it's nice to have around.
template <typename Cont>
concept SubscriptableContainer = Container<Cont> && requires(Cont v, Cont const cv, typename Cont::value_type val, int i) {
    v[i];
    cv[i];
    v.at(i);
    cv.at(i);
};

/**
 * @concept ContainerOrInitializer
 *
 * Checks that a type satisfies the Container requirement or is an inizializer list.
 */
template <typename T>
concept ContainerOrInitializer = requires(T a, T b, T const ca, T const cb, T &ra, T &rb, T const &rca, T const &rcb, T &&rra, T &&rrb) {
    // Check that the types exist.
    typename T::value_type;
    typename T::reference;
    typename T::const_reference;
    typename T::iterator;
    typename T::const_iterator;
    typename T::size_type;

    // Check the properties of the types.
    requires std::forward_iterator<typename T::iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::iterator>, typename T::value_type>;
    requires std::is_convertible_v<typename T::iterator, typename T::const_iterator>;
    requires std::forward_iterator<typename T::const_iterator>;
    requires std::is_same_v<std::iter_value_t<typename T::const_iterator>, typename T::value_type>;
    requires std::is_integral_v<typename T::size_type>;
    requires std::is_unsigned_v<typename T::size_type>;

    // Expressions.
    T();
    { a.begin() } -> std::same_as<typename T::iterator>;
    { ca.begin() } -> std::same_as<typename T::const_iterator>;
    { a.end() } -> std::same_as<typename T::iterator>;
    { ca.end() } -> std::same_as<typename T::const_iterator>;

    { a.size() } -> std::same_as<typename T::size_type>;
    { ca.size() } -> std::same_as<typename T::size_type>;
    { a.max_size() } -> std::same_as<typename T::size_type>;
};

} // namespace einsums