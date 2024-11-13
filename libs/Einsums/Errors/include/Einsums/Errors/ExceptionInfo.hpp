//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ErrorCode.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace einsums {

template <typename Tag, typename Type>
struct ErrorInfo {
    using tag  = Tag;
    using type = Type;

    explicit ErrorInfo(Type const &value) : _value(value) {}

    explicit ErrorInfo(Type &&value) : _value(std::move(value)) {}

    Type _value;
};

#define EINSUMS_DEFINE_ERROR_INFO(NAME, TYPE)                                                                                              \
    struct NAME : ::einsums::ErrorInfo<NAME, TYPE> {                                                                                       \
        explicit NAME(TYPE const &value) : ErrorInfo(value) {                                                                              \
        }                                                                                                                                  \
                                                                                                                                           \
        explicit NAME(TYPE &&value) : ErrorInfo(std::forward<TYPE>(value)) {                                                               \
        }                                                                                                                                  \
    } /**/

namespace detail {

struct ExceptionInfoNodeBase {
    virtual ~ExceptionInfoNodeBase() = default;

    [[nodiscard]] virtual auto lookup(std::type_info const &tag) const noexcept -> void const * = 0;

    std::shared_ptr<ExceptionInfoNodeBase> next;
};

template <typename... Ts>
struct ExceptionInfoNode : public ExceptionInfoNodeBase, Ts... {
    template <typename... ErrorInfo>
    explicit ExceptionInfoNode(ErrorInfo &&...tagged_values) : Ts(tagged_values)... {}

    [[nodiscard]] auto lookup(std::type_info const &tag) const noexcept -> void const * override {
        using entry_type = std::pair<std::type_info const &, void const *>;

        entry_type const entries[] = {{typeid(typename Ts::tag), std::addressof(static_cast<Ts const *>(this)->_value)}...};

        for (auto const &entry : entries) {
            if (entry.first == tag) {
                return entry.second;
            }
        }

        return next ? next->lookup(tag) : nullptr;
    }

    using ExceptionInfoNodeBase::next;
};

} // namespace detail

struct ExceptionInfo {
    using node_ptr = std::shared_ptr<detail::ExceptionInfoNodeBase>;

    ExceptionInfo() noexcept : _data(nullptr) {}

    ExceptionInfo(ExceptionInfo const &other) noexcept = default;
    ExceptionInfo(ExceptionInfo &&other) noexcept      = default;

    auto operator=(ExceptionInfo const &other) noexcept -> ExceptionInfo & = default;
    auto operator=(ExceptionInfo &&other) noexcept -> ExceptionInfo      & = default;

    virtual ~ExceptionInfo() = default;

    template <typename... ErrorInfo>
    auto set(ErrorInfo &&...tagged_values) -> ExceptionInfo & {
        using node_type = detail::ExceptionInfoNode<ErrorInfo...>;

        node_ptr node = std::make_shared<node_type>(std::forward<ErrorInfo>(tagged_values)...);
        node->next    = std::move(_data);
        _data         = std::move(node);
        return *this;
    }

    template <typename Tag>
    auto get() const noexcept -> typename Tag::type const * {
        auto const *data = _data.get();
        return static_cast<typename Tag::type const *>(data ? data->lookup(typeid(typename Tag::tag)) : nullptr);
    }

  private:
    node_ptr _data;
};

namespace detail {

struct ExceptionWithInfoBase : public ExceptionInfo {
    ExceptionWithInfoBase(std::type_info const &type, ExceptionInfo xi) : ExceptionInfo(std::move(xi)), type(type) {}

    std::type_info const &type;
};

template <typename E>
struct ExceptionWithInfo : E, ExceptionWithInfoBase {
    explicit ExceptionWithInfo(E const &e, ExceptionInfo xi) : E(e), ExceptionWithInfoBase(typeid(E), std::move(xi)) {}

    explicit ExceptionWithInfo(E &&e, ExceptionInfo xi) : E(std::move(e)), ExceptionWithInfoBase(typeid(E), std::move(xi)) {}
};

} // namespace detail

template <typename E>
[[noreturn]] void throw_with_info(E &&e, ExceptionInfo &&xi = ExceptionInfo()) {
    using ed = std::decay_t<E>;
    static_assert(std::is_class_v<ed> && !std::is_final_v<ed>, "E shall be a valid base class");
    static_assert(!std::is_base_of_v<ExceptionInfo, ed>, "E shall not derive from exception_info");

    throw detail::ExceptionWithInfo<ed>(std::forward<E>(e), std::move(xi));
}

template <typename E>
[[noreturn]] void throw_with_info(E &&e, ExceptionInfo const &xi) {
    throw_with_info(std::forward<E>(e), ExceptionInfo(xi));
}

///////////////////////////////////////////////////////////////////////////
template <typename E>
auto get_exception_info(E &e) -> ExceptionInfo * {
    return dynamic_cast<ExceptionInfo *>(std::addressof(e));
}

template <typename E>
auto get_exception_info(E const &e) -> ExceptionInfo const * {
    return dynamic_cast<ExceptionInfo const *>(std::addressof(e));
}

///////////////////////////////////////////////////////////////////////////
template <typename E, typename F>
auto invoke_with_exception_info(E const &e, F &&f) -> decltype(std::forward<F>(f)(std::declval<ExceptionInfo const *>())) {
    return std::forward<F>(f)(dynamic_cast<ExceptionInfo const *>(std::addressof(e)));
}

template <typename F>
auto invoke_with_exception_info(std::exception_ptr const &p, F &&f) -> decltype(std::forward<F>(f)(std::declval<ExceptionInfo const *>())) {
    try {
        if (p) {
            std::rethrow_exception(p);
        }
    } catch (ExceptionInfo const &xi) {
        return std::forward<F>(f)(&xi);
    } catch (std::exception const &e) {
        return fmt::format("std::exception:\n  what():  {}", e.what());
    } catch (...) {
    }
    return std::forward<F>(f)(nullptr);
}

template <typename F>
auto invoke_with_exception_info(ErrorCode const &ec, F &&f) -> decltype(std::forward<F>(f)(std::declval<ExceptionInfo const *>())) {
    return invoke_with_exception_info(detail::access_exception(ec), std::forward<F>(f));
}

} // namespace einsums