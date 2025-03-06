#include <Einsums/Config/Types.hpp>

#include <memory>

namespace einsums {

EINSUMS_SINGLETON_IMPL(GlobalConfigMap)

GlobalConfigMap::GlobalConfigMap()
    : str_map_{ConfigMap<std::string>::create()}, int_map_{ConfigMap<std::int64_t>::create()}, double_map_{ConfigMap<double>::create()} {
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

std::string const &GlobalConfigMap::get_string(std::string const &key) const {
    return str_map_->get_value().at(key);
}
std::int64_t GlobalConfigMap::get_int(std::string const &key) const {
    return int_map_->get_value().at(key);
}
double GlobalConfigMap::get_double(std::string const &key) const {
    return double_map_->get_value().at(key);
}
bool GlobalConfigMap::get_bool(std::string const &key) const {
    return bool_map_->get_value().at(key);
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

} // namespace einsums