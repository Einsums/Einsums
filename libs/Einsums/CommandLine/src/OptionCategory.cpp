#include <Einsums/CommandLine/OptionCategory.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

namespace einsums::cl {

AbstractOptionCategory::AbstractOptionCategory(std::string const &help) : help_ { help } {
}

AbstractOptionCategory::AbstractOptionCategory(PrivateConstructorStuff) : AbstractOptionCategory() {
}

AbstractOptionCategory::AbstractOptionCategory(PrivateConstructorStuff, std::string const &help) : AbstractOptionCategory(help) {
}

void AbstractOptionCategory::add_option(std::shared_ptr<Option> option) {
    for (auto curr_opt : this->options_) {
        if (option->long_name() == curr_opt->long_name()) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Can not add option with name '{}'! Option with that name already exists.",
                    option->long_name());
        }
    }
    this->options_.push_back(option);
}

void AbstractOptionCategory::remove_option(std::shared_ptr<Option const> option) {
    this->options_.remove_if([&](auto opt) -> bool {
        return opt == option;
    });
}

void AbstractOptionCategory::remove_option(std::string const &name) {
    this->options_.remove_if([&](auto opt) -> bool {
        return opt->long_name() == name;
    });
}

std::shared_ptr<Option> AbstractOptionCategory::get_option(std::string const &name) {
    for (auto curr_opt : this->options_) {
        if (curr_opt->long_name() == name) {
            return curr_opt;
        }
    }

    EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find the requested option!");
}

std::shared_ptr<Option const> AbstractOptionCategory::get_option(std::string const &name) const {
    for (auto const curr_opt : this->options_) {
        if (curr_opt->long_name() == name) {
            return curr_opt;
        }
    }

    EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find the requested option!");
}

bool AbstractOptionCategory::validate() const {

    std::string error_string;

    error_string.reserve(1024);
    for (auto const &option : this->options_) {
        if (!option->validate(error_string)) {
            return false;
        }
    }
    return true;
}

void AbstractOptionCategory::print_help_line(std::string const &program_name) const {
    std::puts(this->get_name().c_str());
    std::puts(this->help_.c_str());

    for (auto option : this->options_) {
        option->print_help_line();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

PositionalCategory::PositionalCategory(PrivateConstructorStuff) : PositionalCategory() {
}

void PositionalCategory::add_subcategory(std::shared_ptr<AbstractOptionCategory> category) {
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Can not create a subcategory of positional arguments!");
}

void PositionalCategory::remove_subcategory(std::shared_ptr<AbstractOptionCategory const> category) {
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Positional arguments can't have subcategories!");
}

std::shared_ptr<AbstractOptionCategory> PositionalCategory::get_subcategory_by_name(std::string const &name) {
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Positional arguments can't have subcategories!");
}

std::shared_ptr<AbstractOptionCategory const> PositionalCategory::get_subcategory_by_name(std::string const &name) const {
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Positional arguments can't have subcategories!");
}

void PositionalCategory::process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) {
    // First, check to see if the arguments are from a file. This code path shouldn't happen.
    if (from_config) {
        // Now, check to see if all of the arguments can be ignored. We can't take positional arguments
        // from a file.

        if (args.size() > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Can not accept positional arguments from a file!");
        }

        for (auto opt : this->options_) {
            if (opt->occurrence() == Occurrence::Required || opt->occurrence() == Occurrence::OneOrMore) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Can not accept positional arguments from a file!");
            }
        }
    } else {
        // Loop over arguments and have the options process them.

        auto option_it = this->options_.begin();
        auto arg_it = args.cbegin();
        std::string error_string = "";

        while (option_it != this->options_.end() && arg_it != args.cend()) {
            bool success = (*option_it)->parse_argument(std::get<0>(*arg_it), std::get<1>(*arg_it), error_string, from_config);

            if (!success) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", error_string);
            }

            if ((*option_it)->occurrence() != Occurrence::ZeroOrMore || (*option_it)->occurrence() != Occurrence::OneOrMore) {
                option_it++;
            }
            arg_it++;
        }

        // Now check for stragglers.
        if (arg_it != args.cend()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many positional arguments!");
        }

        // Now check for required arguments that weren't set.
        if (option_it != this->options_.end()) {
            option_it++;
        }

        while (option_it != this->options_.end()) {
            if ((*option_it)->occurrence() == Occurrence::Required || (*option_it)->occurrence() == Occurrence::OneOrMore) {
                EINSUMS_THROW_EXCEPTION(not_enough_args, "Required positional arguments not assigned!");
            }
            option_it++;
        }
    }
}

std::string const& PositionalCategory::get_name() const {
    static std::string name { "Positional Arguments" };

    return name;
}

void PositionalCategory::print_usage() const {
    for (auto const &opt : this->options_) {
        opt->print_usage();
    }
}

///////////////////////////////////////////////////////////////////////

OptionCategory::OptionCategory(std::string const &name) : name_ { name } {
}

OptionCategory::OptionCategory(std::string const &name, std::string const &help) : AbstractOptionCategory(help), name_ { name } {
}

OptionCategory::OptionCategory(PrivateConstructorStuff) : OptionCategory() {
}

OptionCategory::OptionCategory(PrivateConstructorStuff, std::string const &name) : OptionCategory(name) {
}

OptionCategory::OptionCategory(PrivateConstructorStuff, std::string const &name, std::string const &help) : OptionCategory(name, help) {
}

void OptionCategory::add_subcategory(std::shared_ptr<AbstractOptionCategory> category) {
    this->subcategories_[category->get_name()] = category;
}

void OptionCategory::remove_subcategory(std::shared_ptr<AbstractOptionCategory const> category) {
    this->subcategories_.erase(category->get_name());
}

std::shared_ptr<AbstractOptionCategory> OptionCategory::get_subcategory_by_name(std::string const &name) {
    return this->subcategories_.at(name);
}

std::shared_ptr<AbstractOptionCategory const> OptionCategory::get_subcategory_by_name(std::string const &name) const {
    return this->subcategories_.at(name);
}

void OptionCategory::process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) {
    // Loop over the passed options.
    std::string error_string;

    error_string.reserve(1024);

    for (auto &[key, value] : args) {
        bool processed = false;
        for (auto option : this->options_) {
            if (option->long_name() == key) {
                // We'll use temp_option to store the error string here.
                bool error = option->parse_argument(key, value, error_string, from_config);

                if (error) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", error_string);
                }
                processed = true;
                break;
            }
        }
        if (!processed) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Did not recognize option '{}'!", key);
        }
    }
}

bool OptionCategory::validate() const {

    std::string error_string;

    error_string.reserve(1024);

    for (auto const &option : this->options_) {
        if (!option->validate(error_string)) {
            return false;
        }
    }

    for (auto const &[key, value] : this->subcategories_) {
        if (!value->validate()) {
            return false;
        }
    }

    return true;
}

std::string const &OptionCategory::get_name() const {
    return this->name_;
}

/////////////////////////////////////////////////////////////////////////

ExclusiveCategory::ExclusiveCategory(std::string const &name) : name_ { name } {
}

ExclusiveCategory::ExclusiveCategory(std::string const &name, std::string const &help) : AbstractOptionCategory(help), name_ { name } {
}

ExclusiveCategory::ExclusiveCategory(PrivateConstructorStuff) : ExclusiveCategory() {
}

ExclusiveCategory::ExclusiveCategory(PrivateConstructorStuff, std::string const &name) : ExclusiveCategory(name) {
}

ExclusiveCategory::ExclusiveCategory(PrivateConstructorStuff, std::string const &name, std::string const &help) : ExclusiveCategory(name,
        help) {
}

void ExclusiveCategory::process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) {
    // Loop over the passed options.
    std::string error_string;

    error_string.reserve(1024);
    bool processed_one = false;

    for (auto &[key, value] : args) {
        bool processed = false;
        for (auto option : this->options_) {
            if (option->long_name() == key) {
                // We'll use temp_option to store the error string here.
                bool error = option->parse_argument(key, value, error_string, from_config);

                if (error) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", error_string);
                }
                processed = true;
                if (processed_one) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                            "Argument '{}' can not be used in conjunction with some other arguments on the command line!", key);
                }
                processed_one = true;
                break;
            }
        }
        if (!processed) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Did not recognize option '{}'!", key);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

EINSUMS_SINGLETON_IMPL(Registry);

Registry::Registry(PrivateConstructorStuff) : Registry() {
}

Registry::~Registry() {
    all_options_.clear();
    positional_.reset();
    subcategories_.clear();
}

void Registry::add_subcategory(std::shared_ptr<AbstractOptionCategory> category) {
    this->subcategories_[category->get_name()] = category;
}

void Registry::remove_subcategory(std::shared_ptr<AbstractOptionCategory const> category) {
    this->subcategories_.erase(category->get_name());
}

std::shared_ptr<AbstractOptionCategory> Registry::get_subcategory_by_name(std::string const &name) {
    return this->subcategories_.at(name);
}

std::shared_ptr<AbstractOptionCategory const> Registry::get_subcategory_by_name(std::string const &name) const {
    return this->subcategories_.at(name);
}

void Registry::process_options(std::list<std::pair<std::string, std::string>> const &args, bool from_config) {
    // First, go through the start of the arguments to look for positional arguments.
    auto positional_it = args.begin();

    while (positional_it != args.end() && std::get<0>(*positional_it) == "") {
        positional_it++;
    }

    if (positional_ && !from_config) {
        std::list<std::pair<std::string, std::string>> positional_args { args.begin(), positional_it };

        positional_->process_options(args, false);
    }

    // Loop over the passed options.
    std::string error_string;

    error_string.reserve(1024);

    while (positional_it != args.end()) {
        auto const &[key, value] = *positional_it;
        positional_it++;
        bool processed = false;
        for (auto option : this->all_options_) {
            if (option->long_name() == key) {
                // We'll use temp_option to store the error string here.
                bool error = option->parse_argument(key, value, error_string, from_config);

                if (error) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", error_string);
                }
                processed = true;
                break;
            }
        }
        if (!processed) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Did not recognize option '{}'!", key);
        }
    }
}

bool Registry::validate() const {
    std::string error_string;

    error_string.reserve(1024);

    for (auto const &option : this->options_) {
        if (!option->validate(error_string)) {
            return false;
        }
    }

    for (auto const &[key, value] : this->subcategories_) {
        if (!value->validate()) {
            return false;
        }
    }

    return true;
}

void Registry::add_option_to_all(std::shared_ptr<Option> option) {
    all_options_.push_back(option);
}

void Registry::remove_option_from_all(std::shared_ptr<Option const> option) {
    this->all_options_.remove_if([&](auto opt) -> bool {
        return opt == option;
    });
}

void Registry::remove_option_from_all(std::string const &name) {
    this->all_options_.remove_if([&](auto opt) -> bool {
        return opt->long_name() == name;
    });
}

void Registry::add_description(std::string const &description) {
    this->description_ = description;
}

void Registry::print_help_line(std::string const &program) const {
    std::puts("Usage:");

    // Print special cases.
    std::printf("%s -h|--help|-?\n", program.c_str());
    std::printf("%s -v|--version\n", program.c_str());

    // Print general case.
    std::printf(program.c_str());

    // Print the positional arguments first.
    positional_->print_usage();

    // Then print the other options.
    for (auto option : this->all_options_) {
        option->print_usage();
    }

    // Two new lines.
    std::puts("\n");

    std::puts(this->description_.c_str());

    std::puts("\n");

    this->positional_->print_help_line(program);

    for (auto const &[key, value] : this->subcategories_) {
        std::printf("%s:\n", key);

        value->print_help_line(program);
    }
}

std::string const &Registry::get_name() const {
    return name_;
}

//////////////////////////////////////////////////////////////////////////////

EINSUMS_SINGLETON_IMPL(OptionCategoryFactory);


std::shared_ptr<PositionalCategory> OptionCategoryFactory::create_positional_category() {
    auto out = std::make_shared<PositionalCategory>(PositionalCategory::PrivateConstructorStuff{});

    auto &registry = Registry::get_singleton();

    registry.add_subcategory(out);

    return out;
}
}
