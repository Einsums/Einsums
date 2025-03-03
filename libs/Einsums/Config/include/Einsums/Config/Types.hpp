//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/TypeSupport/Observable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace einsums {

/**
 * @class ConfigMap
 *
 * @brief Holds a mapping of string keys to configuration values.
 *
 * Objects of this type can hold maps of configuration variables. They can also act as a subject,
 * which can attach observers. When a configuration variable is updated, this map will notify its
 * observers with the new information. it has all of the methods and typedefs available from std::map.
 *
 * @tparam Value The type of data to be associated with each key.
 */
template <typename Value>
class ConfigMap : public std::enable_shared_from_this<ConfigMap<Value>>, public Observable<std::unordered_map<std::string, Value>> {
  private:
    /**
     * @class PrivateType
     *
     * @brief This class allows for a public constructor that can't be used in public contexts.
     *
     * This class helps users to make shared pointers from this class.
     */
    class PrivateType {
      public:
        explicit PrivateType() = default;
    };

  public:
    /**
     * @typedef MappingType
     *
     * @brief Represents the type used to hold the option map.
     */
    using MappingType = std::unordered_map<std::string, Value>;

    /**
     * Public constructor that can only be accessed in private contexts. Used to make shared pointers
     * from this class.
     */
    ConfigMap(PrivateType) : Observable<std::unordered_map<std::string, Value>>() {}

    /**
     * @brief Create a shared pointer from this class.
     *
     * @return A shared pointer to a ConfigMap.
     */
    static std::shared_ptr<ConfigMap<Value>> create() { return std::make_shared<ConfigMap<Value>>(PrivateType()); }

  private:
    /**
     * @brief Default constructor.
     */
    explicit ConfigMap() = default;

    friend class GlobalConfigMap;
};

/**
 * @typedef SharedConfigMap
 *
 * @brief Shared pointer to a ConfigMap.
 */
template <typename Value>
using SharedConfigMap = std::shared_ptr<ConfigMap<Value>>;
// using SharedInfoMap = std::shared_ptr<InfoMap>;

/**
 * @class GlobalConfigMap
 *
 * @brief This is a map that holds global configuration variables.
 *
 * This map holds three ConfigMap's inside. It has one for each of integer values, floating point values,
 * and string values. Observers can observe this map, and depending on the type of the observer, it will
 * be attached to the appropriate sub-map. This class is a singleton.
 */
class EINSUMS_EXPORT GlobalConfigMap {
    EINSUMS_SINGLETON_DEF(GlobalConfigMap)
  public:
    /**
     * @brief Checks to see if the map is empty.
     */
    bool empty() const noexcept;

    /**
     * @brief Gets the size of the map.
     */
    size_t size() const noexcept;

    /**
     * @brief Gets the maximum number of buckets in the map.
     */
    size_t max_size() const noexcept;

    /**
     * @brief Get the string value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::string const &get_string(std::string const &key) const;

    /**
     * @brief Get the integer value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    std::int64_t get_int(std::string const &key) const;

    /**
     * @brief Get the floating point value stored at the given key.
     *
     * Throws an error if the key is not in the map.
     *
     * @param key The key to query.
     */
    double get_double(std::string const &key) const;

    /**
     * @brief Returns the map containing string options.
     */
    std::shared_ptr<ConfigMap<std::string>> get_string_map();

    /**
     * @brief Returns the map containing integer options.
     */
    std::shared_ptr<ConfigMap<std::int64_t>> get_int_map();

    /**
     * @brief Returns the map containing floating point options.
     */
    std::shared_ptr<ConfigMap<double>> get_double_map();

    /**
     * @brief Attach an observer to the global configuration map.
     *
     * The observer should be an object derived from ConfigObserver. The template parameter
     * on the ConfigObserver class
     * determines which map or maps the observer will be attached to. The template parameter can
     * be either @c std::string , @c std::int64_t , or @c double . If the observer derives from
     * multiple of these observers, it will be attached to each map that it is able to.
     *
     * @param obs The observer to attach.
     */
    template <typename T>
    void attach(T &obs) {
        if constexpr (std::is_convertible_v<std::function<void(std::unordered_map<std::string, std::string> const &)>, T>) {
            str_map_->attach(obs);
        }

        if constexpr (std::is_convertible_v<std::function<void(std::unordered_map<std::string, std::int64_t> const &)>, T>) {
            int_map_->attach(obs);
        }

        if constexpr (std::is_convertible_v<std::function<void(std::unordered_map<std::string, double> const &)>, T>) {
            double_map_->attach(obs);
        }
    }

    /**
     * @brief Detach an observer from the global configuration map.
     *
     * @param obs The observer to remove.
     */
    template <typename T>
    void detach(T &obs) {
        if constexpr (std::is_convertible_v<std::function<void(std::unordered_map<std::string, std::string> const &)>, T>) {
            str_map_->detach(obs);
        }

        if constexpr (std::is_convertible_v<std::function<void(std::unordered_map<std::string, std::int64_t> const &)>, T>) {
            int_map_->detach(obs);
        }

        if constexpr (std::is_convertible_v<std::function<void(std::unordered_map<std::string, double> const &)>, T>) {
            double_map_->detach(obs);
        }
    }

  private:
    explicit GlobalConfigMap();

    /**
     * @property str_map_
     *
     * @brief Holds the string valued options.
     */
    std::shared_ptr<ConfigMap<std::string>> str_map_;

    /**
     * @property int_map_
     *
     * @brief Holds the integer valued options.
     */
    std::shared_ptr<ConfigMap<std::int64_t>> int_map_;

    /**
     * @property double_map_
     *
     * @brief Holds the floating-point valued options.
     */
    std::shared_ptr<ConfigMap<double>> double_map_;
};

} // namespace einsums

template <class Value>
bool operator==(std::unordered_map<std::string, Value> const &lhs, einsums::ConfigMap<Value> const &rhs) {
    return lhs == rhs.get_value();
}

template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, std::unordered_map<std::string, Value> const &rhs) {
    return lhs.get_value() == rhs;
}

template <class Value>
bool operator==(einsums::ConfigMap<Value> const &lhs, einsums::ConfigMap<Value> const &rhs) {
    return lhs.get_value() == rhs.get_value();
}
