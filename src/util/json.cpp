#include "json.hpp"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace avm::util {

class JsonParser {
public:
    explicit JsonParser(std::string_view s) : s_(s) {}

    Json parse_document() {
        skip_ws();
        Json v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size()) fail("unexpected trailing characters");
        return v;
    }

private:
    static constexpr int kMaxDepth = 64;

    [[noreturn]] void fail(const std::string& msg) const {
        std::size_t line = 1, col = 1;
        for (std::size_t i = 0; i < pos_ && i < s_.size(); ++i) {
            if (s_[i] == '\n') { ++line; col = 1; } else { ++col; }
        }
        throw JsonError("JSON parse error at line " + std::to_string(line) +
                        ", column " + std::to_string(col) + ": " + msg);
    }

    bool eof() const { return pos_ >= s_.size(); }
    char peek() const { return eof() ? '\0' : s_[pos_]; }

    void skip_ws() {
        while (!eof() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
                          s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    bool consume_literal(const char* lit) {
        std::size_t n = std::strlen(lit);
        if (s_.compare(pos_, n, lit) == 0) { pos_ += n; return true; }
        return false;
    }

    Json parse_value(int depth) {
        if (depth > kMaxDepth) fail("nesting too deep");
        skip_ws();
        if (eof()) fail("unexpected end of input");
        char c = peek();
        Json v;
        if (c == '{') return parse_object(depth);
        if (c == '[') return parse_array(depth);
        if (c == '"') { v.type_ = Json::Type::String; v.string_ = parse_string(); return v; }
        if (c == 't') { if (!consume_literal("true"))  fail("invalid literal"); v.type_ = Json::Type::Bool; v.bool_ = true;  return v; }
        if (c == 'f') { if (!consume_literal("false")) fail("invalid literal"); v.type_ = Json::Type::Bool; v.bool_ = false; return v; }
        if (c == 'n') { if (!consume_literal("null"))  fail("invalid literal"); return v; }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail(std::string("unexpected character '") + c + "'");
    }

    Json parse_number() {
        std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid number");
        while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        if (peek() == '.') {
            ++pos_;
            if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid number");
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid number");
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        std::string tok(s_.substr(start, pos_ - start));
        Json v;
        v.type_ = Json::Type::Number;
        v.number_ = std::strtod(tok.c_str(), nullptr);
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    unsigned parse_hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = peek();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else fail("invalid \\u escape");
            ++pos_;
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (eof()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') break;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') { out += c; continue; }
            if (eof()) fail("unterminated escape");
            char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {          // surrogate pair
                        if (!(peek() == '\\' && pos_ + 1 < s_.size() && s_[pos_ + 1] == 'u'))
                            fail("lone high surrogate");
                        pos_ += 2;
                        unsigned lo = parse_hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("invalid escape");
            }
        }
        return out;
    }

    Json parse_array(int depth) {
        expect('[');
        Json v;
        v.type_ = Json::Type::Array;
        skip_ws();
        if (peek() == ']') { ++pos_; return v; }
        while (true) {
            v.values_.push_back(parse_value(depth + 1));
            skip_ws();
            if (peek() == ',') { ++pos_; continue; }
            expect(']');
            break;
        }
        return v;
    }

    Json parse_object(int depth) {
        expect('{');
        Json v;
        v.type_ = Json::Type::Object;
        skip_ws();
        if (peek() == '}') { ++pos_; return v; }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("expected string key");
            std::string key = parse_string();
            skip_ws();
            expect(':');
            Json val = parse_value(depth + 1);
            v.keys_.push_back(std::move(key));
            v.values_.push_back(std::move(val));
            skip_ws();
            if (peek() == ',') { ++pos_; continue; }
            expect('}');
            break;
        }
        return v;
    }

    std::string_view s_;
    std::size_t pos_{0};
};

Json Json::parse(std::string_view text) {
    return JsonParser(text).parse_document();
}

bool Json::try_parse(std::string_view text, Json& out, std::string* err) {
    try {
        out = parse(text);
        return true;
    } catch (const JsonError& e) {
        if (err) *err = e.what();
        return false;
    }
}

const Json* Json::find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    // Last duplicate key wins.
    for (std::size_t i = keys_.size(); i-- > 0;) {
        if (keys_[i] == key) return &values_[i];
    }
    return nullptr;
}

double Json::number_or(const std::string& key, double def) const {
    const Json* v = find(key);
    return (v && v->is_number()) ? v->number_ : def;
}

bool Json::bool_or(const std::string& key, bool def) const {
    const Json* v = find(key);
    return (v && v->is_bool()) ? v->bool_ : def;
}

std::string Json::string_or(const std::string& key, const std::string& def) const {
    const Json* v = find(key);
    return (v && v->is_string()) ? v->string_ : def;
}

} // namespace avm::util
