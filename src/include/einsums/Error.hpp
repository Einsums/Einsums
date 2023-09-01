/*
 * Copyright (c) 2022 Justin Turney
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <iterator>
#include <variant>
#include <utility>

#include "einsums/_Common.hpp"

namespace einsums {

struct Empty {};

// struct Error {
//     EINSUMS_ALWAYS_INLINE      Error(Error &&)                = default;
//     EINSUMS_ALWAYS_INLINE auto operator=(Error &&) -> Error & = default;
// };

template <typename T, typename E>
struct [[nodiscard]] ErrorOr {
  private:
    template <typename U, typename F>
    friend class ErrorOr;

  public:
    using ResultType = T;
    using ErrorType  = E;

    ErrorOr() = default;

    ErrorOr(ErrorOr &&) noexcept                     = default;
    auto operator=(ErrorOr &&) noexcept -> ErrorOr & = default;

    ErrorOr(ErrorOr const &)                     = delete;
    auto operator=(ErrorOr const &) -> ErrorOr & = delete;

    template <typename U>
    ErrorOr(ErrorOr<U, ErrorType> &&value) :
        _value_or_error(std::move(value._value_or_error)) {}

    template <typename U>
    ErrorOr(U &&value) : _value_or_error(std::forward<U>(value)) {}

    auto value() -> ResultType & {
        return std::get<ResultType>(_value_or_error); }
    auto value() const -> ResultType const & {
        return std::get<ResultType>(_value_or_error); }

    auto error() -> ErrorType & { return std::get<ErrorType>(_value_or_error); }
    auto error() const -> ErrorType const & {
        return std::get<ErrorType>(_value_or_error); }

    [[nodiscard]] auto is_error() const -> bool {
        try {
            std::get<ErrorType>(_value_or_error);
            return true;
        } catch (const std::bad_variant_access &ex) {
            return false;
        }
    }

  private:
    std::variant<ResultType, ErrorType> _value_or_error;
};

template <typename ErrorType>
struct [[nodiscard]] ErrorOr<void, ErrorType> :
        public ErrorOr<Empty, ErrorType> {
    using ResultType = void;
    using ErrorOr<Empty, ErrorType>::ErrorOr;
};

}  // namespace einsums

// using einsums::ErrorOr;
