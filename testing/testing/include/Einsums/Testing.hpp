//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#if defined(_WIN32) && !defined(CATCH_CONFIG_WINDOWS_SEH)
#define CATCH_CONFIG_WINDOWS_SEH
#endif

#include <Einsums/Concepts/Complex.hpp>

#include <catch2/catch_all.hpp>

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
void CheckWithinRel(TestType const &value, TestType const &reference, RemoveComplexT<TestType> const &tolerance) {
    if constexpr (!IsComplexV<TestType>) {
        CHECK_THAT(value, Catch::Matchers::WithinRel(reference, tolerance));
    } else {
        using RealType = RemoveComplexT<TestType>;
        CHECK_THAT(value.real(), Catch::Matchers::WithinRel(reference.real(), RealType{tolerance}));
        CHECK_THAT(value.imag(), Catch::Matchers::WithinRel(reference.imag(), RealType{tolerance}));
    }
}

} // namespace einsums