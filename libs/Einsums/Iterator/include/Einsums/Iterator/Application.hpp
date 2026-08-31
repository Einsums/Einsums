//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <utility>

namespace einsums {

namespace detail {

template<typename Callable, typename Cont, size_t... Idx>
requires(Container<std::remove_cv_t<Cont>>)
inline auto apply(Callable &func, Cont &cont, std::index_sequence<Idx...>) {
    return func(cont[Idx]...);
}

}

template<size_t Elements, typename Callable, typename Cont>
requires(Container<std::remove_cv_t<Cont>>)
inline auto apply(Callable &func, Cont &cont) {
    if(cont.size() != Elements) {
        EINSUMS_THROW_EXCEPTION(num_argument_error, "Container does not have the right number of elements!");
    }
    return einsums::detail::apply(func, cont, std::make_index_sequence<Elements>());
}


}
