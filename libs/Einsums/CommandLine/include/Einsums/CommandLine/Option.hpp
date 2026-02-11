//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace einsums::cl {

enum struct Visibility : uint8_t { Normal, Hidden };
enum struct Occurrence : uint8_t { Optional, Required, ZeroOrMore, OneOrMore };
enum struct ValueExpected : uint8_t { ValueDisallowed, ValueOptional, ValueRequired };


/**
 * @class Option
 *
 * @brief Represents an abstract option.
 *
 * Options are not defined to be positional. They can only be positional if they are in the
 * Positional category.
 */
class Option {
protected:

    /**
     *  The constructors are private so that they can only be accessed by the OptionFactory class.
     *  This way, we can ensure that they are created as shared_ptrs.
     */
    Option() = default;

    Option(std::string long_name, std::initializer_list<char> shorts, std::string help);

    Option(std::string long_name, std::string help);

    friend class OptionFactory;
public:

    virtual ~Option();

    virtual bool parse_argument(std::string const &argstring, std::string val, std::string &err, bool from_config = false) = 0;

    virtual void print_help_line() const = 0;

    virtual void print_usage() const = 0;

    virtual bool validate(std::string &error) const;

    std::string const &long_name() const;

    std::vector<char> const &short_name() const;

    std::string const &help() const;

    Visibility visibility() const;

    Occurrence occurrence() const;

    ValueExpected value_expected() const;

    bool seen_cli() const;

    bool seen_config() const;

    int occurrences() const;

    void on_seen() const;

protected:
    std::string long_name_;
    std::vector<char> short_names_;
    std::string help_;

    Visibility visibility_{Visibility::Normal};
    Occurrence occurrence_{Occurrence::Optional};
    ValueExpected value_expected_{ValueExpected::ValueOptional};

    bool seen_cli_{false};
    bool seen_config_{false};
    int occurrences_{0};

    std::function<void()> on_seen_;
};



}
