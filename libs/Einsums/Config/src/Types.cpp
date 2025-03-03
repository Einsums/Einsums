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

std::shared_ptr<ConfigMap<std::string>> GlobalConfigMap::get_string_map() {
    return str_map_;
}
std::shared_ptr<ConfigMap<std::int64_t>> GlobalConfigMap::get_int_map() {
    return int_map_;
}
std::shared_ptr<ConfigMap<double>> GlobalConfigMap::get_double_map() {
    return double_map_;
}

} // namespace einsums