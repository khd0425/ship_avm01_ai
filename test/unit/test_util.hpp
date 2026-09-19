#pragma once

// Tiny dependency-free test helpers (no gtest needed on the Jetson).

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace avm_test {
inline int& failures() { static int n = 0; return n; }
inline int& checks() { static int n = 0; return n; }
}

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++avm_test::checks();                                                         \
        if (!(cond)) {                                                                \
            ++avm_test::failures();                                                   \
            std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                             \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                         \
    do {                                                                              \
        ++avm_test::checks();                                                         \
        const double _a = static_cast<double>(a), _b = static_cast<double>(b);        \
        if (!(std::fabs(_a - _b) <= static_cast<double>(eps))) {                      \
            ++avm_test::failures();                                                   \
            std::fprintf(stderr, "  FAIL %s:%d  %s=%.9g vs %s=%.9g (eps %g)\n",       \
                         __FILE__, __LINE__, #a, _a, #b, _b, static_cast<double>(eps)); \
        }                                                                             \
    } while (0)

#define TEST_MAIN_END(name)                                                           \
    std::printf("%s: %d checks, %d failed\n", name, avm_test::checks(), avm_test::failures()); \
    return avm_test::failures() == 0 ? 0 : 1
