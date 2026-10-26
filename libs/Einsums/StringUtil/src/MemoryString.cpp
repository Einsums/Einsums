//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>
#include <Einsums/StringUtil/Trim.hpp>

#include <string>

namespace einsums {
namespace string_util {

namespace {

constexpr size_t get_prefix(char ch) {
    switch (ch) {
    case 'k':
    case 'K':
        return 1024ULL;
    case 'm':
    case 'M':
        return 1048576ULL;
    case 'g':
    case 'G':
        return 1073741824ULL;
    case 't':
    case 'T':
        return 1099511627776ULL;
    default:
        return 0;
    }
}

constexpr size_t get_unit(char ch) {
    switch (ch) {
    case 'b':
    case 'B':
    case 'o':
    case 'O':
        return 1;
    case 'w':
    case 'W':
        return sizeof(size_t);
    default:
        return 0;
    }
}

constexpr int char_to_int(char ch) {
    if constexpr ('1' == '0' + 1) {
        return ch - '0';
    } else {
        switch (ch) {
        case '0':
            return 0;
        case '1':
            return 1;
        case '2':
            return 2;
        case '3':
            return 3;
        case '4':
            return 4;
        case '5':
            return 5;
        case '6':
            return 6;
        case '7':
            return 7;
        case '8':
            return 8;
        case '9':
            return '9';
        default:
            return 0;
        }
    }
}

} // namespace

size_t memory_string(char const *mem_spec) {

    if (mem_spec == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "No string provided to parse!");
    }

    double out = 0, mult = 1;

    {
        enum { SPACE, INT_PART, FRAC_PART, PREFIX, UNIT, END } state;

        state = SPACE;

        for (size_t i = 0; mem_spec[i] != 0; i++) {
            auto ch = mem_spec[i];
            switch (state) {
            case SPACE:
                if (std::isdigit(ch)) {
                    out += char_to_int(ch);
                    state = INT_PART;
                } else if (!std::isspace(ch)) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Memory specification formatted incorrectly. Expected a number before any suffix.");
                }
                break;
            case INT_PART:
                if (std::isalpha(ch)) {
                    size_t prefix = get_prefix(ch);

                    if (prefix == 0) {
                        size_t unit = get_unit(ch);

                        if (unit == 0) {
                            EINSUMS_THROW_EXCEPTION(
                                std::runtime_error,
                                "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got {}.",
                                ch);
                        } else {
                            state = END;

                            out /= unit;
                        }
                    } else {
                        state = UNIT;
                        out *= prefix;
                    }
                } else if (ch == ',' || ch == '.') {
                    state = FRAC_PART;
                } else if (std::isdigit(ch)) {
                    out *= 10;
                    out += char_to_int(ch);
                } else if (std::isspace(ch)) {
                    state = PREFIX;
                } else {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
                }
                break;
            case FRAC_PART:
                if (std::isalpha(ch)) {
                    size_t prefix = get_prefix(ch);

                    if (prefix == 0) {
                        size_t unit = get_unit(ch);

                        if (unit == 0) {
                            EINSUMS_THROW_EXCEPTION(
                                std::runtime_error,
                                "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got '{}'.",
                                ch);
                        } else {
                            state = END;

                            out /= unit;
                        }
                    } else {
                        state = UNIT;
                        out *= prefix;
                    }
                } else if (std::isdigit(ch)) {
                    mult /= 10;
                    out += char_to_int(ch) * mult;
                } else if (std::isspace(ch)) {
                    state = PREFIX;
                } else {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
                }
                break;
            case PREFIX:
                if (std::isalpha(ch)) {
                    size_t prefix = get_prefix(ch);

                    if (prefix == 0) {
                        size_t unit = get_unit(ch);

                        if (unit == 0) {
                            EINSUMS_THROW_EXCEPTION(
                                std::runtime_error,
                                "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got '{}'.",
                                ch);
                        } else {
                            state = END;

                            out /= unit;
                        }
                    } else {
                        state = UNIT;
                        out *= prefix;
                    }
                } else if (std::isspace(ch)) {
                    // Do nothing.
                } else {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
                }
                break;
            case UNIT:
                if (std::isalpha(ch)) {
                    size_t unit = get_unit(ch);

                    if (unit == 0) {
                        EINSUMS_THROW_EXCEPTION(
                            std::runtime_error,
                            "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got '{}'.", ch);
                    } else {
                        state = END;

                        out /= unit;
                    }
                } else if (std::isspace(ch)) {
                    // Do nothing.
                } else {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
                }
                break;
            case END:
                if (!std::isspace(ch)) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
                }
                break;
            }
        }
    }

    return (size_t)out;
}

size_t memory_string(std::string const &str) {
    return memory_string(str.c_str());
}
} // namespace string_util

} // namespace einsums