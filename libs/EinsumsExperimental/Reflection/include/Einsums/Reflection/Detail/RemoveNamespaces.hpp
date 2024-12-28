//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TypeSupport/StringLiteral.hpp>

#include <string_view>
#include <utility>

namespace einsums::detail {

template <StringLiteral Name>
consteval auto remove_namespaces() {
    constexpr auto   name = Name.string_view();
    constexpr size_t pos  = name.find_last_of(':');
    if constexpr (pos == std::string_view::npos) {
        return Name;
    }
    constexpr auto substr        = name.substr(pos + 1);
    auto const to_string_literal = [&]<auto... Ns>(std::index_sequence<Ns...>) { return StringLiteral<sizeof...(Ns) + 1>{substr[Ns]...}; };
    return to_string_literal(std::make_index_sequence<substr.size()>{});
}

} // namespace einsums