//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------
#include <Einsums/Assert.hpp>
#include <Einsums/CommandLine/CommandLine.hpp>

#include <string>

namespace einsums::cl {
detail::OptionBaseImpl::~OptionBaseImpl() {
    if (canonical_name_.length() != 0) {
        std::fprintf(stderr, "Finalizing OptionBaseImpl: %s\n", canonical_name_.c_str());
    } else {
        std::fprintf(stderr, "Finalizing OptionBaseImpl.\n");
    }
}

OptionBase::OptionBase(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, OptionCategory *cat)
    : long_name_(longName.get_view()), short_names_(shorts), help_(helpText.get_view()) {
    pointer_to_impl_                  = std::make_shared<detail::OptionBaseImpl>();
    pointer_to_impl_->category_       = cat;
    pointer_to_impl_->canonical_name_ = long_name_;
    Registry::get_singleton().add_option(this);

    EINSUMS_ASSERT_MSG(help_.length() != 0, "Do not add a command line argument without giving it a help statement.");
}

OptionBase::OptionBase(StringRef longName, std::initializer_list<char> shorts, StringRef helpText,
                       std::shared_ptr<detail::OptionBaseImpl> const &impl)
    : long_name_(longName.get_view()), short_names_(shorts), help_(helpText.get_view()), pointer_to_impl_{impl} {
    Registry::get_singleton().add_option(this);
}

OptionBase::OptionBase(StringRef positional_name, Positional, StringRef helpText)
    : long_name_(positional_name.get_view()), help_(helpText.get_view()), is_positional_(true) {
    pointer_to_impl_                  = std::make_shared<detail::OptionBaseImpl>();
    pointer_to_impl_->canonical_name_ = long_name_;
    Registry::get_singleton().add_option(this);
    EINSUMS_ASSERT_MSG(help_.length() != 0, "Do not add a command line argument without giving it a help statement.");
}

OptionBase::~OptionBase() {
    std::fprintf(stderr, "Finalizing OptionBase.\n");
}

bool OptionBase::validate([[maybe_unused]] std::string const &error_prefix) const {
    return true;
}

void OptionBase::finalize_default() {
}

std::string const &OptionBase::get_long_name() const noexcept {
    return long_name_;
}
std::string &OptionBase::get_long_name() noexcept {
    return long_name_;
}

std::string const &OptionBase::get_help() const noexcept {
    return help_;
}
std::string &OptionBase::get_help() noexcept {
    return help_;
}

std::vector<char> const &OptionBase::get_short_names() const noexcept {
    return short_names_;
}
std::vector<char> &OptionBase::get_short_names() noexcept {
    return short_names_;
}

OptionCategory const *OptionBase::get_category() const noexcept {
    return pointer_to_impl_->category_;
}
OptionCategory *OptionBase::get_category() noexcept {
    return pointer_to_impl_->category_;
}

ExclusiveCategory const *OptionBase::get_exlusions() const noexcept {
    return pointer_to_impl_->exclusions_;
}
ExclusiveCategory *OptionBase::get_exclusions() noexcept {
    return pointer_to_impl_->exclusions_;
}

Visibility OptionBase::get_visibility() const noexcept {
    return visibility_;
}

Occurrence OptionBase::get_occurrence() const noexcept {
    return pointer_to_impl_->occurrence_;
}

ValueExpected OptionBase::get_value_expected() const noexcept {
    return value_expected_;
}

bool OptionBase::is_positional() const noexcept {
    return is_positional_;
}

bool OptionBase::is_from_cli() const noexcept {
    return pointer_to_impl_->seen_cli_;
}

bool OptionBase::is_from_config() const noexcept {
    return pointer_to_impl_->seen_config_;
}

int OptionBase::num_occurrences() const noexcept {
    return pointer_to_impl_->occurrences_;
}

void OptionBase::set_long_name(std::string const &name) {
    long_name_ = name;
}

void OptionBase::set_category(OptionCategory *category) noexcept {
    pointer_to_impl_->category_ = category;
}

void OptionBase::set_exclusions(ExclusiveCategory *exclusions) noexcept {
    pointer_to_impl_->exclusions_ = exclusions;
}

void OptionBase::set_visibility(Visibility vis) noexcept {
    visibility_ = vis;
}

void OptionBase::set_occurrence(Occurrence occurrence) noexcept {
    pointer_to_impl_->occurrence_ = occurrence;
}

void OptionBase::set_expected(ValueExpected expected) noexcept {
    value_expected_ = expected;
}

void OptionBase::set_positional(bool positional) noexcept {
    is_positional_ = positional;
}

void OptionBase::seen_on_cli(int count) noexcept {
    if (count > 0) {
        pointer_to_impl_->seen_cli_ = true;
        pointer_to_impl_->occurrences_ += count;
        if (pointer_to_impl_->on_seen_) {
            pointer_to_impl_->on_seen_();
        }
    }
}

void OptionBase::seen_in_config() noexcept {
    pointer_to_impl_->seen_config_ = true;
    if (pointer_to_impl_->on_seen_) {
        pointer_to_impl_->on_seen_();
    }
}

std::shared_ptr<detail::OptionBaseImpl> OptionBase::get_impl() noexcept {
    return pointer_to_impl_;
}

std::shared_ptr<detail::OptionBaseImpl const> OptionBase::get_impl() const noexcept {
    return pointer_to_impl_;
}

Flag &Flag::on_set(std::function<void(bool const &)> f) {
    setter_ = std::move(f);
    return *this;
}

void Flag::finalize_default() {
    if (set_on_unseen_ || num_occurrences() > 0 || is_from_cli() || is_from_config()) {
        set_result(value_);
    }
}

bool Flag::parse_token(std::string_view, std::optional<std::string_view> val, std::string const &error_prefix, bool from_config) {
    bool tmp;
    if (!val.has_value()) {
        tmp = has_implicit_override_ ? implicit_on_ : true; // presence => true
    } else if (!parse_value(*val, tmp, error_prefix)) {
        return false;
    }
    value_ = tmp;
    set_result(value_);
    if (from_config) {
        seen_in_config();
        return true;
    }
    seen_on_cli();
    return true;
}

void Flag::print_help_line(std::string_view, size_t pad_long, size_t pad_short) const {
    if (get_visibility() == Visibility::Hidden)
        return;
    std::string shorts;
    for (char c : get_short_names())
        shorts += einsums::format("-{}, ", c);
    if (!shorts.empty())
        shorts.erase(shorts.end() - 2, shorts.end());
    fmt::print("  --{:<{}}  {:<{}}  {}\n", get_long_name(), pad_long, shorts, pad_short, get_help());
}

bool Flag::get() const noexcept {
    return bound_ ? *bound_ : value_;
}

void Flag::do_set_on_unseen(bool do_it) noexcept {
    set_on_unseen_ = do_it;
}

void Flag::set_value(bool value) noexcept {
    value_ = value;
}

void Flag::set_result(bool value) {
    if (bound_) {
        *bound_ = value;
    }

    if (setter_) {
        setter_(value);
    }
}

void Flag::apply_arg(OptionCategory &c) noexcept {
    set_category(&c);
}
void Flag::apply_arg(Visibility v) noexcept {
    set_visibility(v);
}
void Flag::apply_arg(Occurrence o) noexcept {
    set_occurrence(o);
}
void Flag::apply_arg(Location<bool> loc) noexcept {
    bound_ = loc.ptr;
}
void Flag::apply_arg(std::function<void(bool const &)> f) {
    setter_ = std::move(f);
}
void Flag::apply_arg(DefaultTag<bool> d) noexcept {
    value_ = d.v;
    if (bound_) {
        *bound_ = value_;
    }
}
void Flag::apply_arg(ImplicitValueTag<bool> d) noexcept {
    implicit_on_           = d.v;
    has_implicit_override_ = true;
}
void Flag::apply_arg(ExclusiveCategory &cat) {
    cat.add_option(this);
    set_exclusions(&cat);
}

bool Alias::parse_token(std::string_view, std::optional<std::string_view> val, std::string const &error_prefix, bool from_config) {
    std::optional<std::string_view> v = preset_value_ ? std::optional{std::string_view(*preset_value_)} : val;
    return target_->parse_token(get_impl()->canonical_name_, v, error_prefix, from_config);
}

void Alias::print_help_line(std::string_view, size_t pad_long, size_t pad_short) const {
    if (get_visibility() == Visibility::Hidden) {
        return;
    }
    std::string shorts;
    for (char c : get_short_names()) {
        shorts += einsums::format("-{}, ", c);
    }
    if (!shorts.empty()) {
        shorts.erase(shorts.end() - 2, shorts.end());
    }
    fmt::print("  {:<{}}  {:<{}}  {} (alias for --{})\n", einsums::format("--{}", get_long_name()), pad_long, shorts, pad_short, get_help(),
               get_impl() ? get_impl()->canonical_name_ : "?");
}

void Alias::apply_arg(Visibility v) noexcept {
    set_visibility(v);
}

void Alias::apply_arg(ValueExpected ve) noexcept {
    set_expected(ve);
}

void Alias::apply_arg(std::string const &v) {
    preset_value_ = v;
}

} // namespace einsums::cl