#pragma once

/**
 * @def EINSUMS_SINGLETON_DEF
 *
 * Turns a C++ class into a singleton. Place this at the beginning of the class. You will then need to define a private
 * constructor with no arguments to actually construct the singleton stuff. This macro is only for the definition of the
 * class and does not contain any code. Make sure to use a matching \c EINSUMS_SINGLETON_IMPL somewhere else to 
 * get the code to compile.
 *
 * This will provide the two methods <tt>Type &get_singleton()</tt> and <tt>finalize_singleton()</tt>.
 *
 * @param Type The type of singleton to construct.
 */
#define EINSUMS_SINGLETON_DEF(Type)                                                                                                        \
  private:                                                                                                                                 \
    class PrivateConstructorStuff {};                                                                                                      \
    static Type **get_singleton_pointer();                                                                                                 \
                                                                                                                                           \
  public:                                                                                                                                  \
    Type(PrivateConstructorStuff ignore) : Type() {                                                                                        \
    }                                                                                                                                      \
    static Type &get_singleton();                                                                                                          \
                                                                                                                                           \
    static void finalize_singleton();

/**
 * @def EINSUMS_SINGLETON_IMPL
 *
 * Creates the code for managing a singleton.
 */
#define EINSUMS_SINGLETON_IMPL(Type)                                                                                                       \
    Type **Type::get_singleton_pointer() {                                                                                                 \
        static Type *singleton_instance = nullptr;                                                                                         \
        return &singleton_instance;                                                                                                        \
    }                                                                                                                                      \
                                                                                                                                           \
    Type &Type::get_singleton() {                                                                                                          \
        Type **singleton_instance = get_singleton_pointer();                                                                               \
                                                                                                                                           \
        if (*singleton_instance == nullptr) {                                                                                              \
            *singleton_instance = Type(PrivateConstructorStuff());                                                                         \
        }                                                                                                                                  \
        return **singleton_instance;                                                                                                       \
    }                                                                                                                                      \
                                                                                                                                           \
    void Type::finalize_singleton() {                                                                                                      \
        Type **singleton_instance = get_singleton_pointer();                                                                               \
                                                                                                                                           \
        if (*singleton_instance != nullptr) {                                                                                              \
            delete *singleton_instance;                                                                                                    \
        }                                                                                                                                  \
    }