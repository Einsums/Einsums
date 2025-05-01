//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace einsums::json {

struct JSONWriter {
    struct Entry {
        enum class Type { Object, Array } type_ = Type::Object;
        Entry(Type type = Type::Object) : type_(type) {}

        void write(std::string const &key, std::string_view const &value);
        void write(std::string const &key, char const *value);
        void write(std::string const &key, int value);
        void write(std::string const &key, double value);
        void write(std::string const &key, bool value);
        void write(std::string const &key, std::nullptr_t);

        std::shared_ptr<Entry> object(std::string const &key);
        std::shared_ptr<Entry> array(std::string const &key);
        std::shared_ptr<Entry> object(); // For arrays
        void                   push(std::shared_ptr<Entry> entry);

        std::string to_string(int indent = 0) const;

      private:
        friend class JSONWriter;

        using Value = std::variant<std::string, std::string_view, int, double, bool, std::nullptr_t, std::shared_ptr<Entry>>;

        std::map<std::string, Value> object_data_;
        std::vector<Value>           array_data_;
        mutable std::mutex           mutex_;

        void write_key_value(std::string const &key, Value val);
        void write_array_value(Value val);

        static std::string indent_str(int indent);
        static std::string escape(std::string_view const &str);
        std::string        value_to_string(Value const &val, int indent) const;
    };

    JSONWriter();

    std::shared_ptr<Entry> record(std::string const &name);
    std::string            str(bool pretty = true) const;

  private:
    std::shared_ptr<Entry> root_;
    mutable std::mutex     mutex_;
};

} // namespace einsums::json