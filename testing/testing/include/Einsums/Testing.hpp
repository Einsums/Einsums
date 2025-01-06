//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>

#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "catch2/matchers/catch_matchers_templated.hpp"

#if defined(EINSUMS_WINDOWS)
#    define CATCH_CONFIG_WINDOWS_SEH
#endif
#include <catch2/catch_all.hpp>
#include <type_traits>

namespace einsums {

/**
 * @struct WithinStrictMatcher
 *
 * Catch2 matcher that matches the strictest range for floating point operations.
 */
template <typename T>
struct WithinStrictMatcher : public Catch::Matchers::MatcherGenericBase {};

template <>
struct WithinStrictMatcher<float> : public Catch::Matchers::MatcherGenericBase {
  private:
    float _value, _scale;

  public:
    WithinStrictMatcher(float value, float scale) : _value(value), _scale(scale) {}

    bool match(float other) const {
        // Minimum error is 5.96e-8, according to LAPACK docs.
        if (_value == 0.0f) {
            return std::abs(other) <= 5.960464477539063e-08f * _scale;
        } else {
            return std::abs((other - _value) / _value) <= 5.960464477539063e-08f * _scale;
        }
    }

    std::string describe() const override {
        return "is within a fraction of " + Catch::StringMaker<float>::convert(5.960464477539063e-08f * _scale) + " to " +
               Catch::StringMaker<float>::convert(_value);
    }

    float get_error() const { return 5.960464477539063e-08f * _scale; }
};

template <>
struct WithinStrictMatcher<double> : public Catch::Matchers::MatcherGenericBase {
  private:
    double _value, _scale;

  public:
    WithinStrictMatcher(double value, double scale) : _value(value), _scale(scale) {}

    bool match(double other) const {
        // Minimum error is 1.1e-16, according to LAPACK docs.
        if (_value == 0.0f) {
            return std::abs(other) <= 1.1102230246251565e-16 * _scale;
        } else {
            return std::abs((other - _value) / _value) <= 1.1102230246251565e-16 * _scale;
        }
    }

    std::string describe() const override {
        return "is within a fraction of " + Catch::StringMaker<double>::convert(1.1102230246251565e-16 * _scale) + " to " +
               Catch::StringMaker<double>::convert(_value);
    }

    double get_error() const { return 1.1102230246251565e-16 * _scale; }
};

template <typename T>
auto WithinStrict(T value, T scale = T{1.0}) -> WithinStrictMatcher<T> {
    return WithinStrictMatcher<T>{value, scale};
}

template <typename TestType>
class WithinRelMatcher : public Catch::Matchers::MatcherGenericBase {
  public:
    WithinRelMatcher(TestType value, double eps) : target_{value}, eps_{eps} {}
    bool match(TestType value) const {
        if (target_ == RemoveComplexT<std::remove_cvref_t<TestType>>{0.0}) {
            return std::abs(value) <= eps_;
        } else {
            return std::abs((value - target_) / target_) <= eps_;
        }
    }

    std::string describe() const override {
        if constexpr (IsComplexV<std::remove_cvref_t<TestType>>) {
            return "and " + Catch::StringMaker<RemoveComplexT<std::remove_cvref_t<TestType>>>::convert(target_.real()) + ((target_.imag() < 0) ? "-" : "+") +
                   Catch::StringMaker<RemoveComplexT<std::remove_cvref_t<TestType>>>::convert(std::abs(target_.imag())) + "i are within " +
                   Catch::StringMaker<double>::convert(eps_ * 100) + "% of each other";
        } else {
            return "and " + Catch::StringMaker<std::remove_cvref_t<TestType>>::convert(target_) + " are within " +
                   Catch::StringMaker<double>::convert(eps_ * 100) + "% of each other";
        }
    }

  private:
    TestType target_;
    double   eps_;
};

#ifdef __cpp_deduction_guides
template <typename TestType>
WithinRelMatcher(TestType, double) -> WithinRelMatcher<TestType>;
#endif

template <typename TestType>
WithinRelMatcher<std::remove_cvref_t<TestType>> CheckWithinRel(TestType reference, double tolerance) {
    return WithinRelMatcher(reference, tolerance);
}

} // namespace einsums