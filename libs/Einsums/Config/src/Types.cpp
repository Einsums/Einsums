//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>

#include <memory>

namespace einsums {

EINSUMS_SINGLETON_IMPL(GlobalConfigMap)

GlobalConfigMap::GlobalConfigMap()
    : str_map_{ConfigMap<std::string>::create()}, int_map_{ConfigMap<std::int64_t>::create()}, double_map_{ConfigMap<double>::create()},
      bool_map_{ConfigMap<bool>::create()} {
}

bool GlobalConfigMap::empty() const noexcept {
    return str_map_->get_value().empty() && int_map_->get_value().empty() && double_map_->get_value().empty();
}

size_t GlobalConfigMap::size() const noexcept {
    return str_map_->get_value().size() + int_map_->get_value().size() + double_map_->get_value().size();
}

size_t GlobalConfigMap::max_size() const noexcept {
    return str_map_->get_value().max_size() + int_map_->get_value().max_size() + double_map_->get_value().max_size();
}

std::string const &GlobalConfigMap::get_string(std::string const &key, std::string const &default_value) const {
    if (str_map_->get_value().contains(key)) {
        return str_map_->get_value().at(key);
    } else {
        return default_value;
    }
}
std::int64_t GlobalConfigMap::get_int(std::string const &key, std::int64_t default_value) const {
    if (int_map_->get_value().contains(key)) {
        return int_map_->get_value().at(key);
    } else {
        return default_value;
    }
}
double GlobalConfigMap::get_double(std::string const &key, double default_value) const {
    if (double_map_->get_value().contains(key)) {
        return double_map_->get_value().at(key);
    } else {
        return default_value;
    }
}
bool GlobalConfigMap::get_bool(std::string const &key, bool default_value) const {
    if (bool_map_->get_value().contains(key)) {
        return bool_map_->get_value().at(key);
    } else {
        return default_value;
    }
}

void GlobalConfigMap::set_string(std::string const &key, std::string const &value) {
    str_map_->get_value()[key] = value;
}

void GlobalConfigMap::set_int(std::string const &key, std::int64_t value) {
    int_map_->get_value()[key] = value;
}

void GlobalConfigMap::set_double(std::string const &key, double value) {
    double_map_->get_value()[key] = value;
}

void GlobalConfigMap::set_bool(std::string const &key, bool value) {
    bool_map_->get_value()[key] = value;
}

std::shared_ptr<ConfigMap<std::string>> GlobalConfigMap::get_string_map() {
    return str_map_;
}
std::shared_ptr<ConfigMap<std::int64_t>> GlobalConfigMap::get_int_map() {
    return int_map_;
}
std::shared_ptr<ConfigMap<double>> GlobalConfigMap::get_double_map() {
    return double_map_;
}
std::shared_ptr<ConfigMap<bool>> GlobalConfigMap::get_bool_map() {
    return bool_map_;
}

bool GlobalConfigMap::has_key(std::string const &key, GlobalConfigMap::KeyMembership *key_member_of) const {
    uint8_t out_value = 0;

    if (str_map_->get_value().contains(key)) {
        out_value |= GlobalConfigMap::String;
    }

    if (int_map_->get_value().contains(key)) {
        out_value |= GlobalConfigMap::Int;
    }

    if (double_map_->get_value().contains(key)) {
        out_value |= GlobalConfigMap::Double;
    }

    if (bool_map_->get_value().contains(key)) {
        out_value |= GlobalConfigMap::Bool;
    }

    if (key_member_of != nullptr) {
        *key_member_of = static_cast<GlobalConfigMap::KeyMembership>(out_value);
    }

    return static_cast<bool>(out_value);
}

bool GlobalConfigMap::key_in_strings(std::string const &key) const {
    return str_map_->get_value().contains(key);
}

bool GlobalConfigMap::key_in_ints(std::string const &key) const {
    return int_map_->get_value().contains(key);
}

bool GlobalConfigMap::key_in_doubles(std::string const &key) const {
    return double_map_->get_value().contains(key);
}

bool GlobalConfigMap::key_in_bools(std::string const &key) const {
    return bool_map_->get_value().contains(key);
}

size_t einsums::hashes::insensitive_hash<std::string>::operator()(std::string const &str) const noexcept {
    size_t hash = 0;

    // Calculate the mask. If size_t is N bytes, mask for the top N bits.
    // The first part creates a Mersenne value with the appropriate number of bits.
    // The second shifts it to the top.
    constexpr size_t mask = (((size_t)1 << sizeof(size_t)) - 1) << (7 * sizeof(size_t));

    for (char ch : str) {
        char upper = std::toupper(ch);
        if (upper == '-') { // Convert dashes to underscores.
            upper = '_';
        }

        hash <<= sizeof(size_t); // Shift left a number of bits equal to the number of bytes in size_t.
        hash += (uint8_t)upper;

        if ((hash & mask) != (size_t)0) {
            hash ^= mask >> (6 * sizeof(size_t));
            hash &= ~mask;
        }
    }
    return hash;
}

size_t einsums::hashes::insensitive_hash<char *>::operator()(char const *str) const noexcept {
    size_t hash = 0;

    // Calculate the mask. If size_t is N bytes, mask for the top N bits.
    // The first part creates a Mersenne value with the appropriate number of bits.
    // The second shifts it to the top.
    constexpr size_t mask       = (((size_t)1 << sizeof(size_t)) - 1) << (7 * sizeof(size_t));
    size_t           curr_index = 0;

    while (str[curr_index] != 0) {
        char upper = std::toupper(str[curr_index]);
        if (upper == '-') { // Convert dashes to underscores.
            upper = '_';
        }

        hash <<= sizeof(size_t); // Shift left a number of bits equal to the number of bytes in size_t.
        hash += (uint8_t)upper;

        if ((hash & mask) != (size_t)0) {
            hash ^= mask >> (6 * sizeof(size_t));
            hash &= ~mask;
        }
        curr_index++;
    }
    return hash;
}

void GlobalConfigMap::lock() {
    str_map_->lock();
    int_map_->lock();
    double_map_->lock();
    bool_map_->lock();
}

bool GlobalConfigMap::try_lock() {
    bool res = str_map_->try_lock();

    if (!res) {
        return false;
    }

    res = int_map_->try_lock();

    if (!res) {
        str_map_->unlock();
        return false;
    }

    res = double_map_->try_lock();

    if (!res) {
        str_map_->unlock();
        int_map_->unlock();
        return false;
    }

    res = bool_map_->try_lock();

    if (!res) {
        str_map_->unlock();
        int_map_->unlock();
        double_map_->unlock();
        return false;
    }

    return true;
}

void GlobalConfigMap::unlock(bool notify) {
    str_map_->unlock(notify);
    int_map_->unlock(notify);
    double_map_->unlock(notify);
    bool_map_->unlock(notify);
}

} // namespace einsums