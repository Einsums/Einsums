//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>
#include <Einsums/CommandLine/Option.hpp>

#include <unordered_map>
#include <memory>
#include <string>

namespace einsums::cl {

/**
 * @class AbstractOptionCategory
 *
 * @brief Represents an abstract category of options.
 *
 * This uses the Composite design pattern, but each level can also store options.
 */
class EINSUMS_EXPORT AbstractOptionCategory {
protected:
    /**
     *  This constructor is essentially private. This makes it so that objects of this type
     *  can only be created using the OptionCategoryFactory.
     */
    AbstractOptionCategory() = default;

    AbstractOptionCategory(std::string const &help);

    struct PrivateConstructorStuff {
    };

    friend class OptionCategoryFactory;

public:
    // Now we delete the other special constructors.
    AbstractOptionCategory(AbstractOptionCategory const&) = delete;
    AbstractOptionCategory(AbstractOptionCategory&&) = delete;

    AbstractOptionCategory(PrivateConstructorStuff);

    AbstractOptionCategory(PrivateConstructorStuff, std::string const &help);

    virtual ~AbstractOptionCategory() = default;

    virtual void add_subcategory(std::shared_ptr<AbstractOptionCategory> category) = 0;

    virtual void remove_subcategory(std::shared_ptr<AbstractOptionCategory const> category) = 0;

    void add_option(std::shared_ptr<Option> option);

    void remove_option(std::shared_ptr<Option const> option);

    void remove_option(std::string const &name);

    virtual std::shared_ptr<AbstractOptionCategory> get_subcategory_by_name(std::string const &name) = 0;

    virtual std::shared_ptr<AbstractOptionCategory const> get_subcategory_by_name(std::string const &name) const = 0;

    virtual void process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) = 0;

    virtual bool validate() const;

    virtual std::string const& get_name() const = 0;

    std::shared_ptr<Option> get_option(std::string const &name);

    std::shared_ptr<Option const> get_option(std::string const &name) const;

    virtual void print_help_line(std::string const &program_name) const;

protected:
    std::list<std::shared_ptr<Option>> options_;

    std::string help_;
};

/**
 * @class PositionalCategory
 *
 * @brief The category for handling positional options. There should only be one per registry.
 */
class EINSUMS_EXPORT PositionalCategory final : public AbstractOptionCategory {
private:
    PositionalCategory() = default;

    friend class OptionCategoryFactory;

    using AbstractOptionCategory::PrivateConstructorStuff;

public:
    PositionalCategory(PrivateConstructorStuff);

    ~PositionalCategory() = default;

    void add_subcategory(std::shared_ptr<AbstractOptionCategory> category) override;

    void remove_subcategory(std::shared_ptr<AbstractOptionCategory const> category) override;

    std::shared_ptr<AbstractOptionCategory> get_subcategory_by_name(std::string const &name) override;

    std::shared_ptr<AbstractOptionCategory const> get_subcategory_by_name(std::string const &name) const override;

    void process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) override;

    std::string const &get_name() const override;

    void print_usage() const;
};

/**
 *  @class OptionCategory
 *
 *  @brief A concrete option category.
 *
 *  These categories have a name, can have options, and can have subcategories.
 */
class EINSUMS_EXPORT OptionCategory : public AbstractOptionCategory {
protected:
    OptionCategory() = default;

    OptionCategory(std::string const &name);

    OptionCategory(std::string const &name, std::string const &help);

    friend class OptionCategoryFactory;

    using AbstractOptionCategory::PrivateConstructorStuff;

public:
    OptionCategory(PrivateConstructorStuff);

    OptionCategory(PrivateConstructorStuff, std::string const &name);

    OptionCategory(PrivateConstructorStuff, std::string const &name, std::string const &help);

    virtual ~OptionCategory() = default;

    void add_subcategory(std::shared_ptr<AbstractOptionCategory> category) override;

    void remove_subcategory(std::shared_ptr<AbstractOptionCategory const> category) override;

    std::shared_ptr<AbstractOptionCategory> get_subcategory_by_name(std::string const &name) override;

    std::shared_ptr<AbstractOptionCategory const> get_subcategory_by_name(std::string const &name) const override;

    void process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) override;

    bool validate() const override;

    std::string const& get_name() const override;

protected:
    std::unordered_map<std::string, std::shared_ptr<AbstractOptionCategory>> subcategories_;

    std::string name_;
};

class EINSUMS_EXPORT ExclusiveCategory : public AbstractOptionCategory {
protected:
    ExclusiveCategory() = default;

    ExclusiveCategory(std::string const &name);

    ExclusiveCategory(std::string const &name, std::string const &help);

    friend class OptionCategoryFactory;

    using AbstractOptionCategory::PrivateConstructorStuff;

public:

    ExclusiveCategory(PrivateConstructorStuff);

    ExclusiveCategory(PrivateConstructorStuff, std::string const &name);

    ExclusiveCategory(PrivateConstructorStuff, std::string const &name, std::string const &help);

    virtual ~ExclusiveCategory() = default;

    void process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) override;

    bool validate() const override;

    std::string const& get_name() const override;

protected:
    std::string name_;
};

/**
 * @class Registry
 *
 * @brief Top-level option category with no name.
 *
 * To get the positional argument list, either use its singleton method
 */
class EINSUMS_EXPORT Registry final : public AbstractOptionCategory {
    // We could use SINGLETON_DEF here, but we are going to expand it out so that we can
    // reuse private constructor types.
    // We are still able to use EINSUMS_SINGLETON_IMPL, though.
    // EINSUMS_SINGLETON_DEF(Registry);

private:
    Registry() = default;

    using AbstractOptionCategory::PrivateConstructorStuff;

public:

    static Registry &get_singleton();

    Registry(PrivateConstructorStuff);

    ~Registry();

    void add_subcategory(std::shared_ptr<AbstractOptionCategory> category) override;

    void remove_subcategory(std::shared_ptr<AbstractOptionCategory const> category) override;

    std::shared_ptr<AbstractOptionCategory> get_subcategory_by_name(std::string const &name) override;

    std::shared_ptr<AbstractOptionCategory const> get_subcategory_by_name(std::string const &name) const override;

    void process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) override;

    bool validate() const override;

    void add_option_to_all(std::shared_ptr<Option> option);

    void remove_option_from_all(std::shared_ptr<Option const> option);

    void remove_option_from_all(std::string const &name);

    std::shared_ptr<PositionalCategory> get_positional_category();

    std::shared_ptr<PositionalCategory const> get_positional_category() const;

    void add_description(std::string const &description);

    void print_help_line(std::string const &program) const override;

    std::string const &get_name() const override;

private:
    std::unordered_map<std::string, std::shared_ptr<AbstractOptionCategory>> subcategories_;

    std::shared_ptr<PositionalCategory> positional_;

    std::list<std::shared_ptr<Option>> all_options_;

    std::string description_;
    std::string name_{"Top level options"};
};

/**
 *  @class OptionCategoryFactory
 *
 *  Creates objects for the different category types and handles attaching them to a category.
 */
class EINSUMS_EXPORT OptionCategoryFactory final {
    EINSUMS_SINGLETON_DEF(OptionCategoryFactory);
private:
    OptionCategoryFactory() = default;
public:

    template<typename... Args>
    std::shared_ptr<OptionCategory> create_category(Args&&... args) {
        auto out = std::make_shared<OptionCategory>(OptionCategory::PrivateConstructorStuff {}, std::forward<Args>(args)...);

        auto &registry = Registry::get_singleton();

        registry.add_subcategory(out);

        return out;
    }

    template<typename... Args>
    std::shared_ptr<OptionCategory> create_subcategory(std::shared_ptr<AbstractOptionCategory> parent, Args&&... args) {
        auto out = std::make_shared<OptionCategory>(OptionCategory::PrivateConstructorStuff {}, std::forward<Args>(args)...);

        parent->add_subcategory(out);

        return out;
    }

    template<typename... Args>
    std::shared_ptr<ExclusiveCategory> create_exclusive_category(Args&&... args) {
        auto out = std::make_shared<ExclusiveCategory>(ExclusiveCategory::PrivateConstructorStuff {}, std::forward<Args>(args)...);

        auto &registry = Registry::get_singleton();

        registry.add_subcategory(out);

        return out;
    }

    template<typename... Args>
    std::shared_ptr<ExclusiveCategory> create_exclusive_subcategory(std::shared_ptr<AbstractOptionCategory> parent, Args&&... args) {
        auto out = std::make_shared<ExclusiveCategory>(ExclusiveCategory::PrivateConstructorStuff {}, std::forward<Args>(args)...);

        parent->add_subcategory(out);

        return out;
    }

    std::shared_ptr<PositionalCategory> create_positional_category();
};

}
