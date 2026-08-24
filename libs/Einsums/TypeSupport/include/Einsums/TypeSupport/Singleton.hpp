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
 * This will provide the <tt>Type &get_singleton()</tt> method..
 *
 * @param Type The type of singleton to construct.
 */
#define EINSUMS_SINGLETON_DEF(Type)                                                                                                        \
  private:                                                                                                                                 \
    class PrivateConstructorStuff {};                                                                                                      \
                                                                                                                                           \
  public:                                                                                                                                  \
    Type(PrivateConstructorStuff ignore) : Type() {                                                                                        \
    }                                                                                                                                      \
    static Type &get_singleton();                                                                                                          \
    Type(const Type &) = delete;                                                                                                           \
    Type(Type &&)      = delete;

/**
 * @def EINSUMS_SINGLETON_IMPL
 *
 * Creates the code for managing a singleton.
 */
#define EINSUMS_SINGLETON_IMPL(Type)                                                                                                       \
    Type &Type::get_singleton() {                                                                                                          \
        static std::unique_ptr<Type> singleton_instance = std::make_unique<Type>(PrivateConstructorStuff());                               \
        return *singleton_instance;                                                                                                        \
    }

/**
 * @def EINSUMS_THREAD_MULTITON_DEF
 *
 * Turns a C++ class into a multiton with one instance per thread.. Place this at the beginning of the class. You will then need to define a
 * private constructor with no arguments to actually construct the singleton stuff. This macro is only for the definition of the class and
 * does not contain any code. Make sure to use a matching \c EINSUMS_THREAD_MULTITON_IMPL somewhere else to get the code to compile.
 *
 * This will provide the <tt>Type &get_thread_instance()</tt> method.
 *
 * @param Type The type of multiton to construct.
 */
#define EINSUMS_THREAD_MULTITON_DEF(Type)                                                                                                  \
  private:                                                                                                                                 \
    class PrivateConstructorStuff {};                                                                                                      \
                                                                                                                                           \
  public:                                                                                                                                  \
    Type(PrivateConstructorStuff ignore) : Type() {                                                                                        \
    }                                                                                                                                      \
    static Type &get_thread_instance();                                                                                                    \
    Type(const Type &) = delete;                                                                                                           \
    Type(Type &&)      = delete;

/**
 * @def EINSUMS_THREAD_MULTITON_IMPL
 *
 * Creates the code for managing a multiton.
 */
#define EINSUMS_THREAD_MULTITON_IMPL(Type)                                                                                                 \
    Type &Type::get_singleton() {                                                                                                          \
        thread_local std::unique_ptr<Type> thread_instance = std::make_unique<Type>(PrivateConstructorStuff());                            \
        return *thread_instance;                                                                                                           \
    }
