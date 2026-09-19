#pragma once

// Minimal JSON DOM (parse + read accessors). No external dependency so the
// host-side modules build and unit-test on any machine.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace avm::util {

class JsonError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;

    /// Parse a JSON document. Throws JsonError with line/column on failure.
    static Json parse(std::string_view text);

    /// Non-throwing variant; returns false and fills `err` on failure.
    static bool try_parse(std::string_view text, Json& out, std::string* err = nullptr);

    Type type() const { return type_; }
    bool is_null()   const { return type_ == Type::Null; }
    bool is_bool()   const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array()  const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    // Value accessors return `def` when the type does not match.
    bool as_bool(bool def = false) const { return is_bool() ? bool_ : def; }
    double as_number(double def = 0.0) const { return is_number() ? number_ : def; }
    const std::string& as_string() const { return string_; }
    std::string as_string(const std::string& def) const { return is_string() ? string_ : def; }

    // Array access
    const std::vector<Json>& items() const { return values_; }
    std::size_t size() const { return values_.size(); }

    // Object access (insertion-ordered). `find` returns nullptr if absent.
    const Json* find(const std::string& key) const;
    const std::vector<std::string>& keys() const { return keys_; }

    // Convenience "get with default" for objects.
    double number_or(const std::string& key, double def) const;
    bool bool_or(const std::string& key, bool def) const;
    std::string string_or(const std::string& key, const std::string& def) const;

private:
    friend class JsonParser;

    Type type_{Type::Null};
    bool bool_{false};
    double number_{0.0};
    std::string string_;
    std::vector<std::string> keys_;   // object keys (parallel to values_)
    std::vector<Json> values_;        // array items or object values
};

} // namespace avm::util
