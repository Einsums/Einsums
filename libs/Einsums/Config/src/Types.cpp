#include <Einsums/Config/Types.hpp>

#include <memory>

namespace einsums {

EINSUMS_SINGLETON_IMPL(GlobalConfigMap)

GlobalConfigMap::GlobalConfigMap()
    : str_map_{ConfigMap<std::string>::create()}, int_map_{ConfigMap<std::int64_t>::create()}, double_map_{ConfigMap<double>::create()} {
}

bool GlobalConfigMap::empty() const noexcept {
    return str_map_->empty() && int_map_->empty() && double_map_->empty();
}

size_t GlobalConfigMap::size() const noexcept {
    return str_map_->size() + int_map_->size() + double_map_->size();
}

size_t GlobalConfigMap::max_size() const noexcept {
    return str_map_->max_size() + int_map_->max_size() + double_map_->max_size();
}

void GlobalConfigMap::clear() noexcept {
    str_map_->clear();
    int_map_->clear();
    double_map_->clear();
}

size_t GlobalConfigMap::erase(std::string const &key) {
    size_t ret = 0;
    if (str_map_->contains(key)) {
        ret += str_map_->erase(key);
    }
    if (int_map_->contains(key)) {
        ret += int_map_->erase(key);
    }
    if (double_map_->contains(key)) {
        ret += double_map_->erase(key);
    }

    return ret;
}

std::string &GlobalConfigMap::at_string(std::string const &key) {
    return str_map_->at(key);
}
std::int64_t &GlobalConfigMap::at_int(std::string const &key) {
    return int_map_->at(key);
}
double &GlobalConfigMap::at_double(std::string const &key) {
    return double_map_->at(key);
}

std::string const &GlobalConfigMap::at_string(std::string const &key) const {
    return str_map_->at(key);
}
std::int64_t const &GlobalConfigMap::at_int(std::string const &key) const {
    return int_map_->at(key);
}
double const &GlobalConfigMap::at_double(std::string const &key) const {
    return double_map_->at(key);
}

std::string &GlobalConfigMap::get_string(std::string const &key) {
    return str_map_->operator[](key);
}
std::int64_t &GlobalConfigMap::get_int(std::string const &key) {
    return int_map_->operator[](key);
}
double &GlobalConfigMap::get_double(std::string const &key) {
    return double_map_->operator[](key);
}

std::string const &GlobalConfigMap::get_string(std::string const &key) const {
    return str_map_->operator[](key);
}
std::int64_t const &GlobalConfigMap::get_int(std::string const &key) const {
    return int_map_->operator[](key);
}
double const &GlobalConfigMap::get_double(std::string const &key) const {
    return double_map_->operator[](key);
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

void GlobalConfigMap::notify() {
    str_map_->notify();
    int_map_->notify();
    double_map_->notify();
}

} // namespace einsums