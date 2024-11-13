#pragma once

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace einsums::detail {

template <typename T, typename Alloc>
struct WeakVector;

template <typename T, typename Alloc = std::allocator<T>, typename Deleter = std::default_delete<std::vector<T, Alloc>>>
using UniqueVector = std::unique_ptr<std::vector<T, Alloc>, Deleter>;

template <typename T, typename Alloc = std::allocator<T>>
struct SharedVector {
  public:
    using value_type             = typename std::vector<T, Alloc>::value_type;
    using allocator_type         = typename std::vector<T, Alloc>::allocator_type;
    using size_type              = typename std::vector<T, Alloc>::size_type;
    using difference_type        = typename std::vector<T, Alloc>::difference_type;
    using reference              = typename std::vector<T, Alloc>::reference;
    using const_reference        = typename std::vector<T, Alloc>::const_reference;
    using pointer                = typename std::vector<T, Alloc>::pointer;
    using const_pointer          = typename std::vector<T, Alloc>::const_pointer;
    using iterator               = typename std::vector<T, Alloc>::iterator;
    using const_iterator         = typename std::vector<T, Alloc>::const_iterator;
    using reverse_iterator       = typename std::vector<T, Alloc>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<T, Alloc>::const_reverse_iterator;

    /*
     * Shared pointer definitions and declarations.
     */
    constexpr SharedVector() noexcept : _base() {}
    constexpr SharedVector(std::nullptr_t null) noexcept : _base(null) {}

    template <class Y>
    explicit SharedVector(Y *ptr, size_type offset = 0) : _base(ptr), _offset{offset} {}

    template <class Y, class Deleter>
    SharedVector(Y *ptr, size_type offset, Deleter del) : _base(ptr, del), _offset{offset} {}

    template <class Y, class Deleter>
    SharedVector(Y *ptr, Deleter del) : _base(ptr, del) {}

    template <class Deleter>
    SharedVector(std::nullptr_t ptr, Deleter del) : _base(ptr, del) {}

    template <class Y, class Deleter, class Alloc2>
    SharedVector(Y *ptr, size_type offset, Deleter del, Alloc2 alloc) : _base(ptr, del, alloc), _offset{offset} {}

    template <class Y, class Deleter, class Alloc2>
    SharedVector(Y *ptr, Deleter del, Alloc2 alloc) : _base(ptr, del, alloc) {}

    template <class Deleter, class Alloc2>
    SharedVector(std::nullptr_t ptr, Deleter del, Alloc2 alloc) : _base(ptr, del, alloc) {}

    template <typename Y>
    SharedVector(const std::shared_ptr<Y> &ptr, size_type offset = 0) noexcept : _base(ptr), _offset{offset} {}

    SharedVector(const SharedVector<T, Alloc> &ptr, size_type offset = 0) noexcept : _base(ptr._base), _offset{ptr._offset + offset} {}

    template <class Y>
    SharedVector(std::shared_ptr<Y> &&ptr, size_type offset = 0) noexcept : _base(std::forward(ptr)), _offset{offset} {}

    SharedVector(SharedVector<T, Alloc> &&ptr, size_type offset = 0) noexcept
        : _base(std::move(ptr._base)), _offset{ptr._offset + offset} {}

    template <class Y>
    SharedVector(const std::weak_ptr<Y> &ptr, size_type offset = 0) : _base(ptr), _offset{offset} {}

    SharedVector(const WeakVector<T, Alloc> &ptr, size_type offset = 0) : _base(ptr.get_base()), _offset{ptr.offset() + offset} {}

    template <class Y, class Deleter>
    SharedVector(const std::unique_ptr<Y, Deleter> &ptr, size_type offset = 0) : _base(ptr), _offset{offset} {}

    virtual ~SharedVector() = default;

    SharedVector<T, Alloc> &operator=(const SharedVector<T, Alloc> &ptr) noexcept {
        _base   = ptr._base;
        _offset = ptr._offset;

        return *this;
    }

    template <typename Y>
    SharedVector<T, Alloc> &operator=(const std::shared_ptr<Y> &ptr) noexcept {
        _base   = ptr;
        _offset = 0;

        return *this;
    }

    SharedVector<T, Alloc> &operator=(SharedVector<T, Alloc> &&ptr) noexcept {
        _base   = std::move(ptr._base);
        _offset = ptr._offset;

        return *this;
    }

    template <typename Y>
    SharedVector<T, Alloc> &operator=(std::shared_ptr<Y> &&ptr) noexcept {
        _base   = ptr;
        _offset = 0;

        return *this;
    }

    template <class Y, class Deleter>
    SharedVector<T, Alloc> &operator=(std::unique_ptr<Y, Deleter> &&ptr) {
        _base   = ptr;
        _offset = 0;

        return *this;
    }

    void reset() noexcept {
        _base.reset();
        _offset = 0;
    }

    template <class Y>
    void reset(Y *ptr) {
        _base.reset(ptr);
        _offset = 0;
    }

    template <class Y, class Deleter>
    void reset(Y *ptr, Deleter del) {
        _base.reset(ptr, del);
        _offset = 0;
    }

    template <class Y, class Deleter, class Alloc2>
    void reset(Y *ptr, Deleter del, Alloc2 alloc) {
        _base.reset(ptr, del, alloc);
        _offset = 0;
    }

    void swap(SharedVector<T, Alloc> &ptr) noexcept {
        std::swap(_offset, ptr._offset);
        _base.swap(ptr._base);
    }

    std::vector<T, Alloc> *get() const noexcept { return _base.get(); }

    std::shared_ptr<std::vector<T, Alloc>> get_base() noexcept { return _base; }

    std::vector<T, Alloc> &operator*() const noexcept { return *_base; }

    std::vector<T, Alloc> *operator->() const noexcept { return _base.get(); }

    long use_count() const noexcept { return _base.use_count(); }

    explicit operator bool() const noexcept { return (bool)_base; }

    template <class Y>
    bool owner_before(const std::shared_ptr<Y> &other) const noexcept {
        return _base.owner_before(other);
    }

    bool owner_before(const SharedVector<T, Alloc> &other) const noexcept { return _base.owner_before(other._base); }

    template <class Y>
    bool owner_before(const std::weak_ptr<Y> &other) const noexcept {
        return _base.owner_before(other);
    }

    bool owner_before(const WeakVector<T, Alloc> &other) const noexcept { return _base.owner_before(other.get_base()); }

    /*
     * Vector definitions and declarations.
     */

    explicit SharedVector(const Alloc &alloc) : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(alloc)} {}

    explicit SharedVector(size_type n, const Alloc &alloc = Alloc())
        : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(n, alloc)} {}

    SharedVector(size_type n, const T &val, const Alloc &alloc = Alloc())
        : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(n, val, alloc)} {}

    template <class InputIterator>
    SharedVector(InputIterator first, InputIterator last, const Alloc &alloc = Alloc())
        : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(first, last, alloc)} {}

    SharedVector(const std::vector<T, Alloc> &copy) : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(copy)} {}

    SharedVector(const std::vector<T, Alloc> &copy, const Alloc alloc)
        : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(copy, alloc)} {}

    SharedVector(std::vector<T, Alloc> &&move) : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(std::forward(move))} {}

    SharedVector(std::vector<T, Alloc> &&move, const Alloc alloc)
        : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(std::forward(move), alloc)} {}

    SharedVector(std::initializer_list<T> il, const Alloc alloc = Alloc())
        : _base{std::make_shared<std::shared_ptr<std::vector<T, Alloc>>>(il, alloc)} {}

    constexpr SharedVector &operator=(const std::vector<T, Alloc> &other) {
        if (_offset == 0) {
            *_base = other;
        } else {
            if (_base->size() - _offset < other.size()) {
                _base->resize(_offset + other.size());
            }

            for (size_type element = 0; element < other.size(); element++) {
                (*_base)[element] = other[element];
            }
        }
    }

    constexpr void assign(size_type count, const T &value) {
        if (_base->size() < count + _offset) {
            _base->resize(count + _offset);
        }

        for (auto iter = std::next(_base->begin(), _offset); iter < _base->end(); iter++) {
            *iter = value;
        }
    }

    template <class InputIterator>
    constexpr void assign(InputIterator first, InputIterator last) {
        auto dist = std::distance(first, last);
        if (_base->size() < dist + _offset) {
            _base->resize(dist + _offset);
        }

        for (auto iter1 = std::next(_base->begin(), _offset), iter2 = first; iter1 < _base->end() && iter2 < last; iter1++, iter2++) {
            *iter1 = *iter2;
        }
    }

    constexpr void assign(std::initializer_list<T> list) {
        if (_base->size() < list.size() + _offset) {
            _base.resize(list.size() + _offset);
        }

        for (auto iter1 = std::next(_base->begin(), _offset), iter2 = list.begin(); iter1 < _base->end() && iter2 < list.end();
             iter1++, iter2++) {
            *iter1 = *iter2;
        }
    }

    constexpr allocator_type get_allocator() const noexcept { return _base->get_allocator(); }

    constexpr reference at(size_type pos) { return _base->at(pos + _offset); }

    constexpr const_reference at(size_type pos) const { return _base->at(pos + _offset); }

    constexpr reference operator[](size_type pos) { return (*_base)[pos + _offset]; }

    constexpr const_reference operator[](size_type pos) const { return (*_base)[pos + _offset]; }

    constexpr reference front() { return _base->front(); }

    constexpr const_reference front() const { return _base->front(); }

    constexpr reference back() { return _base->back(); }

    constexpr const_reference back() const { return _base->back(); }

    constexpr T *data() noexcept { return _base->data(); }

    constexpr const T *data() const noexcept { return _base->data(); }

    constexpr iterator begin() noexcept { return std::next(_base->begin(), _offset); }

    constexpr const_iterator begin() const noexcept { return std::next(_base->begin(), _offset); }

    constexpr const_iterator cbegin() const noexcept { return std::next(_base->cbegin(), _offset); }

    constexpr iterator end() noexcept { return _base->end(); }

    constexpr const_iterator end() const noexcept { return _base->end(); }

    constexpr const_iterator cend() const noexcept { return _base->cend(); }

    constexpr iterator rbegin() noexcept { return _base->rbegin(); }

    constexpr const_iterator rbegin() const noexcept { return _base->rbegin(); }

    constexpr const_iterator crbegin() const noexcept { return _base->crbegin(); }

    constexpr iterator rend() noexcept { return std::prev(_base->rend(), _base->size() - _offset); }

    constexpr const_iterator rend() const noexcept { return std::prev(_base->rend(), _base->size() - _offset); }

    constexpr const_iterator crend() const noexcept { return std::prev(_base->crend(), _base->size() - _offset); }

    constexpr bool empty() const noexcept { return _base->empty() || (_base->size() <= _offset); }

    constexpr size_type size() const noexcept {
        if (_offset > _base->size()) {
            return 0;
        } else {
            return _base->size() - _offset;
        }
    }

    void reserve(size_type new_cap) { _base->reserve(new_cap + _offset); }

    constexpr size_type capacity() const noexcept {
        if (_base->capacity() > _offset) {
            return 0;
        } else {
            return _base->capacity() - _offset;
        }
    }

    constexpr void shrink_to_fit() { _base->shrink_to_fit(); }

    constexpr void clear() noexcept {
        if (_offset == 0) {
            _base->clear();
        } else {
            _base->resize(_offset);
        }
    }

    constexpr iterator insert(const_iterator pos, const T &value) {
        if (std::distance(_base->cbegin(), pos) + 1 < _offset) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->insert(pos, value);
    }

    constexpr iterator insert(const_iterator pos, T &&value) {
        if (std::distance(_base->cbegin(), pos) + 1 < _offset) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->insert(pos, value);
    }

    constexpr iterator insert(const_iterator pos, size_type count, const T &value) {
        if (std::distance(_base->cbegin(), pos) + 1 < _offset) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->insert(pos, count, value);
    }

    template <typename InputIterator>
    constexpr iterator insert(const_iterator pos, InputIterator first, InputIterator last) {
        if (std::distance(_base->cbegin(), pos) + 1 < _offset) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->insert(pos, first, last);
    }

    constexpr iterator insert(const_iterator pos, std::initializer_list<T> list) {
        if (std::distance(_base->cbegin(), pos) + 1 < _offset) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->insert(pos, list);
    }

    template <typename... Args>
    constexpr iterator emplace(const_iterator pos, Args &&...args) {
        if (std::distance(_base->cbegin(), pos) + 1 < _offset) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->emplace(pos, std::forward(args)...);
    }

    constexpr iterator erase(const_iterator pos) {
        if (std::distance(_base->cbegin(), pos) + 1 < _offset) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->erase(pos);
    }

    constexpr iterator erase(const_iterator first, const_iterator last) {
        if ((std::distance(_base->cbegin(), first) + 1 < _offset) || (std::distance(_base->cbegin(), last) + 1 < _offset)) {
            throw std::out_of_range("Accessing SharedVector before its range begins!");
        }

        return _base->erase(first, last);
    }

    constexpr void push_back(const T &value) { _base->push_back(value); }

    constexpr void push_back(T &&value) { _base->push_back(value); }

    template <typename... Args>
    constexpr reference emplace_back(Args &&...args) {
        return _base->emplace_back(std::forward(args)...);
    }

    constexpr void pop_back() { _base->pop_back(); }

    constexpr void resize(size_type count) { _base->resize(count + _offset); }

    constexpr void resize(size_type count, const value_type &value) { _base->resize(count + _offset, value); }

    /*
     * Other methods.
     */

    constexpr size_type offset() const {
        return _offset;
    }

  private:
    std::shared_ptr<std::vector<T, Alloc>> _base;

    size_type _offset{0};
};

template <class InputIterator, class Alloc = std::allocator<typename std::iterator_traits<InputIterator>::value_type>>
SharedVector(InputIterator first, InputIterator last,
             Alloc alloc = Alloc()) -> SharedVector<typename std::iterator_traits<InputIterator>::value_type, Alloc>;

template <typename T, typename Alloc>
SharedVector(WeakVector<T, Alloc> &) -> SharedVector<T, Alloc>;

template <typename T, typename Alloc, typename Deleter>
SharedVector(std::unique_ptr<std::vector<T, Alloc>, Deleter> &) -> SharedVector<T, Alloc>;

template <typename T, typename Alloc = std::allocator<T>>
struct WeakVector {
  public:
    using value_type             = typename std::vector<T, Alloc>::value_type;
    using allocator_type         = typename std::vector<T, Alloc>::allocator_type;
    using size_type              = typename std::vector<T, Alloc>::size_type;
    using difference_type        = typename std::vector<T, Alloc>::difference_type;
    using reference              = typename std::vector<T, Alloc>::reference;
    using const_reference        = typename std::vector<T, Alloc>::const_reference;
    using pointer                = typename std::vector<T, Alloc>::pointer;
    using const_pointer          = typename std::vector<T, Alloc>::const_pointer;
    using iterator               = typename std::vector<T, Alloc>::iterator;
    using const_iterator         = typename std::vector<T, Alloc>::const_iterator;
    using reverse_iterator       = typename std::vector<T, Alloc>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<T, Alloc>::const_reverse_iterator;

    constexpr WeakVector() noexcept : _base() {}

    WeakVector(const WeakVector<T, Alloc> &copy, size_type offset = 0) noexcept : _base(copy.get_base()), _offset{copy.offset() + offset} {}

    template<typename Y>
    WeakVector(const std::weak_ptr<Y> &copy, size_type offset = 0) noexcept : _base(copy), _offset{offset} {}

    WeakVector(const SharedVector<T, Alloc> &copy, size_type offset = 0) noexcept : _base(std::move(copy.get_base())), _offset{copy.offset() + offset} {}

    template<typename Y>
    WeakVector(const std::shared_ptr<Y> &copy, size_type offset = 0) noexcept : _base(copy), _offset{offset} {}

    WeakVector(WeakVector<T, Alloc> &&copy, size_type offset = 0) noexcept : _base(std::move(copy.get_base())), _offset{copy.offset() + offset} {}

    template<typename Y>
    WeakVector(std::weak_ptr<Y> &&copy, size_type offset = 0) noexcept : _base(copy), _offset{offset} {}

    virtual ~WeakVector() = default;

    WeakVector &operator=(const WeakVector<T, Alloc> &copy) noexcept {
        _offset = copy.offset();
        _base = copy.get_base();
        return *this;
    }

    template<typename Y>
    WeakVector &operator=(const std::weak_ptr<Y> &copy) noexcept {
        _offset = 0;
        _base = copy;
        return *this;
    }

    WeakVector &operator=(const SharedVector<T, Alloc> &copy) noexcept {
        _offset = copy.offset();
        _base = copy.get_base();
        return *this;
    }

    template<typename Y>
    WeakVector &operator=(const std::shared_ptr<Y> &copy) noexcept {
        _offset = 0;
        _base = copy;
        return *this;
    }

    WeakVector &operator=(WeakVector<T, Alloc> &&copy) noexcept {
        _offset = std::move(copy.offset());
        _base = std::move(copy.get_base());
        return *this;
    }

    template<typename Y>
    WeakVector &operator=(std::weak_ptr<Y> &&copy) noexcept {
        _offset = 0;
        _base = copy;
        return *this;
    }

    void reset() noexcept {
        _offset = 0;
        _base.reset();
    }

    long use_count() const noexcept {
        return _base.use_count();
    }

    bool expired() const noexcept {
        return _base.expired();
    }

    SharedVector<T, Alloc> lock(size_type offset = 0) const noexcept {
        if(expired()) {
            return SharedVector<T, Alloc>();
        } else {
            return SharedVector<T, Alloc>(*this, offset);
        }
    }

    bool owner_before(const WeakVector<T, Alloc> &other) const noexcept {
        return _base.owner_before(other.get_base());
    }

    template<typename Y>
    bool owner_before(const std::weak_ptr<Y> &other) const noexcept {
        return _base.owner_before(other);
    }

    bool owner_before(const SharedVector<T, Alloc> &other) const noexcept {
        return _base.owner_before(other.get_base());
    }

    template<typename Y>
    bool owner_before(const std::shared_ptr<Y> &other) const noexcept {
        return _base.owner_before(other);
    }

  private:
    std::weak_ptr<std::vector<T, Alloc>> _base;

    size_type _offset{0};
};

template<typename T, typename Alloc>
WeakVector(SharedVector<T, Alloc>) -> WeakVector<T, Alloc>;

} // namespace einsums::detail

template <typename T1, typename Alloc1, typename T2, typename Alloc2>
bool operator==(const einsums::detail::SharedVector<T1, Alloc1> &first, const einsums::detail::SharedVector<T2, Alloc2> &second) {
    if (first.size() != second.size()) {
        return false;
    }
    for (const auto [elem1, elem2] : first, second) {
        if (elem1 != elem2) {
            return false;
        }
    }
    return true;
}

template <typename T1, typename Alloc1, typename T2, typename Alloc2>
auto operator<=>(const einsums::detail::SharedVector<T1, Alloc1> &first, const einsums::detail::SharedVector<T2, Alloc2> &second) {
    return std::lexicographical_compare_three_way(first.cbegin(), first.cend(), second.cbegin(), second.cend());
}

template <typename T, typename Alloc, typename U>
constexpr einsums::detail::SharedVector<T, Alloc>::size_type erase(einsums::detail::SharedVector<T, Alloc> &vec, const U &value) {
    auto iter = vec.begin();

    typename einsums::detail::SharedVector<T, Alloc>::size_type erased = 0;

    while (iter < vec.end()) {
        while (*iter == value && iter < vec.end()) {
            vec.erase(iter);
            erased++;
        }

        iter++;
    }

    return erased;
}

template <typename T, typename Alloc, typename U, typename Pred>
constexpr einsums::detail::SharedVector<T, Alloc>::size_type erase_if(einsums::detail::SharedVector<T, Alloc> &vec, Pred pred) {
    auto iter = vec.begin();

    typename einsums::detail::SharedVector<T, Alloc>::size_type erased = 0;

    while (iter < vec.end()) {
        while (pred(*iter) && iter < vec.end()) {
            vec.erase(iter);
            erased++;
        }

        iter++;
    }

    return erased;
}
