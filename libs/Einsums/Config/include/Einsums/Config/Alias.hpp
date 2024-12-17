#pragma once


// Taken from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
#define ALIAS_TEMPLATE_FUNC(highLevelFunction, lowLevelFunction)                                                                       \
    template <typename... Args>                                                                                                            \
    inline auto highLevelFunction(Args &&...args) -> decltype(lowLevelFunction(std::forward<Args>(args)...)) {                             \
        return lowLevelFunction(std::forward<Args>(args)...);                                                                              \
    }