//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @def EINSUMS_SINGLETON_DEF
 *
 * Turns a C++ class into a singleton. Place this at the beginning of the class. You will then need to define a private
 * constructor with no arguments to actually construct the singleton stuff. This macro is only for the definition of the
 * class and does not contain any code. Make sure to use a matching \c EINSUMS_SINGLETON_IMPL somewhere else to
 * get the code to compile.
 *
 * This will provide the <tt>Type &get_singleton()</tt> static method.
 *
 * @param Type The type of singleton to construct.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *     Added the private static method \c get_singleton_ptr and the public static method \c reset_singleton . The former gets the
 *     underlying \c std::unique_ptr that holds the single unique instance, while the latter resets this. On Windows, it is not
 *     recommended to do <tt>get_singleton_ptr().reset(nullptr)</tt> since that may try to call a destructor on a pointer
 *     from another DLL's heap, which will cause a hang up. The \c reset_singleton does this operation in the context of
 *     the same DLL that allocated the pointer in the first place, which should avoid this issue. If you want something to
 *     be able to access the \c get_singleton_ptr method, make it a friend to the singleton class.
 * @endversion
 */
#define EINSUMS_SINGLETON_DEF(Type)                                                                                                        \
  private:                                                                                                                                 \
    class PrivateConstructorStuff {};                                                                                                      \
    static auto get_singleton_ptr() -> std::unique_ptr<Type> &;                                                                            \
                                                                                                                                           \
  public:                                                                                                                                  \
    Type(PrivateConstructorStuff /*ignore*/) : Type() {                                                                                    \
    }                                                                                                                                      \
    static auto get_singleton() -> Type &;                                                                                                 \
    static void reset_singleton();                                                                                                         \
    Type(const Type &) = delete;                                                                                                           \
    Type(Type &&)      = delete;

/**
 * @def EINSUMS_SINGLETON_IMPL
 *
 * Creates the code for managing a singleton.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_SINGLETON_IMPL(Type)                                                                                                       \
    auto Type::get_singleton_ptr() -> std::unique_ptr<Type> & {                                                                            \
        static std::unique_ptr<Type> singleton_instance{nullptr};                                                                          \
        return singleton_instance;                                                                                                         \
    }                                                                                                                                      \
    auto Type::get_singleton() -> Type & {                                                                                                 \
        auto &singleton_instance = get_singleton_ptr();                                                                                    \
        if (!singleton_instance) {                                                                                                         \
            singleton_instance = std::make_unique<Type>(PrivateConstructorStuff());                                                        \
        }                                                                                                                                  \
        return *singleton_instance;                                                                                                        \
    }                                                                                                                                      \
    void Type::reset_singleton() {                                                                                                         \
        auto &singleton_instance = get_singleton_ptr();                                                                                    \
        singleton_instance.reset(nullptr);                                                                                                 \
    }
