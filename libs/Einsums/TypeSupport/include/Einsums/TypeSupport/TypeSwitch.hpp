//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>
namespace einsums {

/**
 * @struct Case<SwitchType, Condition, Result>
 *
 * @brief Represents a case for the type switch types.
 *
 * @tparam Result If the type given matches the condition, then this will be the result.
 * @tparam Condition The type or types to compare against.
 */
template <typename Result, typename... Condition>
struct Case {};

/**
 * @struct Default<Result>
 *
 * @brief Represents the default case for the type switch.
 *
 * @tparam Result The type to return.
 */
template <typename Result>
struct Default {};

/**
 * @struct Switch<switch_type, Cases...>
 *
 * @brief Like a switch statement but for types.
 *
 * Compares the switch type to each of the cases. If the type matches one of the cases, then the resulting type will be
 * the one contained in that case. There can be a default case as well. If the switch reaches the end without figuring out
 * a type, then an error will be thrown.
 *
 * @tparam switch_type The type to check.
 * @tparam Cases The cases to check. There may be a default case at the end as well.
 */
template <typename switch_type, typename... Cases>
struct Switch {
    using type = void;
    void func() {
        static_assert(false, "Switch type was given an unrecognized case! Make sure to use the einsums::Case or einsums::Default types "
                             "when passing to this struct.");
    }
};

#ifndef DOXYGEN
// Case where all cases have failed. This is an error.
template <typename switch_type>
struct Switch<switch_type> {
    using type = void;
    void func() { static_assert(false, "Switch type reached its end without returning a type! Add a default case or check your cases."); }
};

// Case where the first check in the first case fails. Repeat the check on the next check in the first case.
template <typename switch_type, typename Result, typename First, typename... Rest, typename... Cases>
struct Switch<switch_type, Case<Result, First, Rest...>, Cases...> {
    using type = typename Switch<switch_type, Case<Result, Rest...>, Cases...>::type;
};

// Case where the first check in the first case succeeds. Return the result.
template <typename switch_type, typename Result, typename... Rest, typename... Cases>
struct Switch<switch_type, Case<Result, switch_type, Rest...>, Cases...> {
    using type = Result;
};

// Case where all checks in the first case have failed. Continue to the next case.
template <typename switch_type, typename Result, typename... Cases>
struct Switch<switch_type, Case<Result>, Cases...> {
    using type = typename Switch<switch_type, Cases...>::type;
};

// Case where we find a default case, but there are still cases after it. This is an error.
template <typename switch_type, typename Result, typename... Cases>
struct Switch<switch_type, Default<Result>, Cases...> {
    using type = void;
    void func() {
        static_assert(false, "Switch type was given a default case while there are still other cases to process! Re-order your cases so "
                             "that the default case, if present, is last.");
    }
};

// Default case.
template <typename switch_type, typename Result>
struct Switch<switch_type, Default<Result>> {
    using type = Result;
};
#endif

/**
 * @typedef SwitchT<switch_type, Cases...>
 *
 * @brief Like a switch statement but for types.
 *
 * Compares the switch type to each of the cases. If the type matches one of the cases, then the resulting type will be
 * the one contained in that case. There can be a default case as well. If the switch reaches the end without figuring out
 * a type, then an error will be thrown.
 *
 * @tparam switch_type The type to check.
 * @tparam Cases The cases to check. There may be a default case at the end as well.
 */
template <typename... Args>
using SwitchT = typename Switch<Args...>::type;

} // namespace einsums