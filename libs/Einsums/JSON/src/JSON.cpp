//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#include <Einsums/JSON.hpp>

#include <iomanip>
#include <sstream>

namespace einsums::json {

// ------------------------- JSONWriter -------------------------

JSONWriter::JSONWriter() : root_(std::make_shared<Entry>()) {
}

std::shared_ptr<JSONWriter::Entry> JSONWriter::record(std::string const &name) {
    std::lock_guard lock(mutex_);
    auto            entry = std::make_shared<Entry>();
    root_->write_key_value(name, entry);
    return entry;
}

std::string JSONWriter::str(bool pretty) const {
    std::lock_guard lock(mutex_);
    return root_->to_string(pretty ? 0 : -1);
}

// ------------------------- Entry -------------------------

void JSONWriter::Entry::write(std::string const &key, std::string_view const &value) {
    write_key_value(key, escape(value));
}

void JSONWriter::Entry::write(std::string const &key, char const *value) {
    write_key_value(key, std::string_view(value));
}
void JSONWriter::Entry::write(std::string const &key, int value) {
    write_key_value(key, value);
}

void JSONWriter::Entry::write(std::string const &key, double value) {
    write_key_value(key, value);
}

void JSONWriter::Entry::write(std::string const &key, bool value) {
    write_key_value(key, value);
}

void JSONWriter::Entry::write(std::string const &key, std::nullptr_t) {
    write_key_value(key, nullptr);
}

std::shared_ptr<JSONWriter::Entry> JSONWriter::Entry::object(std::string const &key) {
    auto obj = std::make_shared<Entry>();
    write_key_value(key, obj);
    return obj;
}

std::shared_ptr<JSONWriter::Entry> JSONWriter::Entry::array(std::string const &key) {
    auto arr = std::make_shared<Entry>(Type::Array);
    write_key_value(key, arr);
    return arr;
}

std::shared_ptr<JSONWriter::Entry> JSONWriter::Entry::object() {
    auto obj = std::make_shared<Entry>();
    write_array_value(obj);
    return obj;
}

void JSONWriter::Entry::push(std::shared_ptr<Entry> entry) {
    write_array_value(entry);
}

void JSONWriter::Entry::write_key_value(std::string const &key, Value val) {
    std::lock_guard lock(mutex_);
    object_data_[key] = std::move(val);
}

void JSONWriter::Entry::write_array_value(Value val) {
    std::lock_guard lock(mutex_);
    array_data_.emplace_back(std::move(val));
}

std::string JSONWriter::Entry::indent_str(int indent) {
    return std::string(indent, ' ');
}

std::string JSONWriter::Entry::escape(std::string_view const &str) {
    std::ostringstream oss;
    oss << '"';
    for (char c : str) {
        switch (c) {
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            if (c < 0x20)
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
            else
                oss << c;
        }
    }
    oss << '"';
    return oss.str();
}

std::string JSONWriter::Entry::value_to_string(Value const &val, int indent) const {
    if (std::holds_alternative<std::string>(val)) {
        return std::get<std::string>(val);
    }
    if (std::holds_alternative<std::string_view>(val)) {
        return escape(std::get<std::string_view>(val));
    }
    if (std::holds_alternative<int>(val)) {
        return std::to_string(std::get<int>(val));
    }
    if (std::holds_alternative<double>(val)) {
        return std::to_string(std::get<double>(val));
    }
    if (std::holds_alternative<bool>(val)) {
        return std::get<bool>(val) ? "true" : "false";
    }
    if (std::holds_alternative<std::nullptr_t>(val)) {
        return "null";
    }
    if (std::holds_alternative<std::shared_ptr<Entry>>(val)) {
        return std::get<std::shared_ptr<Entry>>(val)->to_string(indent >= 0 ? indent + 2 : -1);
    }
    return "\"<invalid>\"";
}

std::string JSONWriter::Entry::to_string(int indent) const {
    std::lock_guard    lock(mutex_);
    std::ostringstream oss;
    bool               pretty    = indent >= 0;
    std::string        pad       = pretty ? indent_str(indent) : "";
    std::string        newline   = pretty ? "\n" : "";
    std::string        separator = pretty ? ": " : ":";

    if (type_ == Type::Object) {
        oss << "{" << newline;
        bool first = true;
        for (auto const &[key, val] : object_data_) {
            if (!first)
                oss << "," << newline;
            oss << (pretty ? indent_str(indent + 2) : "") << escape(key) << separator << value_to_string(val, indent);
            first = false;
        }
        oss << newline << pad << "}";
    } else {
        oss << "[" << newline;
        bool first = true;
        for (auto const &val : array_data_) {
            if (!first)
                oss << "," << newline;
            oss << (pretty ? indent_str(indent + 2) : "") << value_to_string(val, indent);
            first = false;
        }
        oss << newline << pad << "]";
    }

    return oss.str();
}

} // namespace einsums::json