..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _code_style:

Coding Style
============

When contributing to the Einsums code base, we want to make sure our code fits to certain guidelines.
The main purpose of these guidelines is to get contributors to think more deeply about their
code, and to provide a single style to make our code more consistent to read. All files should begin
with the following statement.

.. code::

  ----------------------------------------------------------------------------------------------
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.
  ----------------------------------------------------------------------------------------------


C++ Style
---------

To make code formatting easier, we have provided two files, ``.clang-tidy`` and ``.clang-format``.
These files can be used with the ``clang-tidy`` package to format C++ code, as well as to hint at
possible errors in your code. One thing that may come up when using these is a strange warning about
certain C++20 features. If you are getting this error, you may need to add a new file called ``.clangd``
which contains the following.

.. code::
    
    CompileFlags:
    #C++20
    Add: [-std=c++20]

Coding Conventions
^^^^^^^^^^^^^^^^^^

Names
"""""

Names should follow these conventions.

* Types, including classes, structs, enums, and typedefs should be in ``PascalCase``.
  This means that every word begins with a capital letter, and words are concatenated
  without any extra symbols. Some examples.

  * ``Tensor``
  * ``GPUView``
  * ``DeviceTensorView``
  * ``Tensor::ValueType``

* Concepts, requirements, and compile-time boolean statements underlying those concepts and requirements
  should also be in ``PascalCase``. Also, requirements and boolean statements should begin
  with ``Is`` and end in ``V``, standing for "value". Some examples.

  * ``TensorConcept`` is a concept.
  * ``IsTensorV`` is a requirement or compile-time boolean statement.

* Namespaces should be in ``snake_case``, where every word is lower case, and each word is
  separated by underscores. Some examples.

  * ``einsums::tensor_algebra``
  * ``einsums::tensor_props``

* Functions and methods should be in ``snake_case`` as well. Some examples.

  * ``einsum(...)``
  * ``A.full_view_of_underlying()``

* Private and protected properties in a class should begin in a single underscore. However,
  some older code will have them begin in a single underscore instead. Some examples.

  * ``this->dims_``: Discouraged for consistency.
  * ``this->_dims``: Preferred.

* Variables and parameters should usually be in ``snake_case``. However, to be consistent with
  mathematical notation, tensor variables and parameters are usually capitalized. This includes
  when the name of a tensor is referenced in a variable name. Some examples.

  * ``std::string const &name``: ``name`` does not refer to a tensor, so it is in snake case.
  * ``TensorType const &A``: ``A`` is a tensor, so it is capitalized.
  * ``std::tuple<AIndices...> const &A_indices``: ``A`` is a tensor, but the variable is not.
    This means that the "A" is capitalized, but the rest of the variable is in snake case.

* Macros and enum members should be in ``ALL_CAPS``.

Order of Things
^^^^^^^^^^^^^^^

* *cv* keywords should never come before the pointer or reference type they modify. As an example,
  
  * ``std::string const &name``, not ``const std::string &name``.
  * However, ``const size_t size`` is allowed.

* When writing a class, the preferred order of blocks is public, then protected, then private.
* Place simple return types in front of function names. Place complicated return types after
  function names. Using the ``auto`` keyword can sometimes mess with Doxygen, so this is the
  compromise we have to make. Some examples.

  * ``size_t size()``
  * ``Tensor<double, 2> &create_tensor()`` or ``auto create_tensor() -> Tensor<double, 2> &``
    are both alright. Use your best jugement here.
  * ``auto common_initialization(TensorType<T, OtherRank> const &other, Args &&...args) -> std::enable_if_t<std::is_base_of_v<::einsums::tensor_base::Tensor<T, OtherRank>, TensorType<T, OtherRank>>>``
    This is a complicated return type. Use the ``auto`` keyword for readability.

Miscellaneous
^^^^^^^^^^^^^

* Indents are four spaces. Do not use tabs.
* Lines should be at most 140 columns long.
* Avoid using the function call operator on a tensor inside a large loop. This is slow. Here are some options, in order of what should be tried.

  * Using :code:`tensor.data()[index]` to directly index a tensor. 
    This is fast, but can only be used on certain kinds of tensors, so be careful. It can be used when iterating over
    a tensor's elements in order. If you are skipping around, it is better to use something else for better readability.
  * Using :cpp:func:`sentinel_to_sentinels` and :code:`tensor.data()[index]`. This should be faster than subscripting tensors,
    but be careful, since it can only be used on certain kinds of tensors. As with the previous method, it can be used when iterating
    over a tensor's elements in order. It is still readable even when skipping around, but it will only work on :cpp:class:`Tensor`s
    and :cpp:class:`TensorView`s.
  * Using :cpp:func:`sentinel_to_indices` and :cpp:func:`subscript_tensor`. This will make the choice between 
    the :code:`subscript` method and the function call syntax, using the :code:`subscript` method as the primary
    and the function call syntax as a fallback. This is the most general way, and it should be preferred unless
    you can ensure specificity in the kinds of tensors you are being passed.

Some constructions need to have serious thought before they are used. Before any code
with these constructions is accepted, their use will need to be justified.

* ``goto`` statements.
* ``do { } while(false);`` blocks outside of macros. They are fine within macros,
  since their use is considered idiomatic to C/C++ for making a macro require a
  semicolon after the closing parenthesis. 
* Inline assembly will be outright banned. One of the goals of Einsums is portability. This goes against
  this goal.
* Anything considered to be undefined behavior. Different compilers and systems may have different
  behavior, so it is best to not use this. Some examples of undefined behavior includes the following.

  * Anything that uses the binary representation of floating point numbers. IEEE 754 states
    that this is only an exchange format. Modifying the underlying binary representation
    is considered to be undefined behavior.
  * Assuming the size of variables. For instance, the presence and size of ``long double`` is highly system dependent.

Python Style
------------

The approach to Python style is to generally follow the standard Python style guidelines. Some things to keep in mind.

* Try to use type annotations when writing Python code. Some examples.
  
  * ``def set_name(name)``: Bad.
  * ``def set_name(name: str)``: Good.
  * ``def iterate_elements(param)``: Fine. ``param`` can be pretty much any type. ``def iterate_elements(param: Any)`` would be preferred,
    but brevity is sometimes better than verbosity.

* Prefer ``PascalCase`` for type names.
* Prefer ``snake_case`` for functions, methods, and variables.
* However, the same considerations for tensor variables apply as in C++. Tensor varaibles are in ``UPPER_CASE``,
  and any reference to a tensor variable in a non-tensor variable should match the case of the tensor.

  * ``A``: Tensor variable. 
  * ``A_indices``: References a tensor variable, but is not a tensor variable.
