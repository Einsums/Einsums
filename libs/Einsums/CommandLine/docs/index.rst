..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_CommandLine:

===================
Einsums CommandLine
===================

A minimal, header‑only C++20 command‑line parser designed in the spirit of LLVM’s
``cl::`` utilities, but simplified and dependency‑free (aside from ``fmt`` for help text).

- **Header‑only** (just include ``Einsums/CommandLine.hpp``)
- **Options**: ``Flag``, ``Opt<T>``, ``List<T>``, ``OptEnum<Enum>``, ``Alias``
- **Named parameter tags**: ``Default(...)``, ``ImplicitValue(...)``, ``ValueName("...")``
- **Bindings**: ``Location<T>`` (write to external storage), ``Setter<T>`` (callback)
- **Config precedence**: *defaults* < *config file/map* < *CLI*
- **Unknown args**: unrecognized options and everything after ``--`` are collected
- **Built‑ins**: ``--help`` and ``--version`` (always available)

Quick Start
-----------

.. code-block:: cpp

   #include <Einsums/CommandLine.hpp>

   using namespace einsums::cl;

   int main(int argc, char** argv) {
     std::vector<std::string> args(argv, argv + argc);

     // 1) Group options into categories (affects help layout)
     OptionCategory General{"General"};

     // 2) Declare options
     Flag Verbose{
       "verbose", {'v'}, "Enable verbose logging",
       General, Default(false), ImplicitValue(true)
     };

     Opt<int> Threads{
       "threads", {'t'},"Number of threads", Default(4),
       General, RangeBetween(1, 256), ValueName("N")
     };

     Opt<std::string> Output{
       "output", {'o'}, "Output file", Default(std::string{"a.out"}),
       General, ValueName("PATH")
     };

     // Positional “gather the rest”
     List<std::string> Inputs{"inputs", Positional{}, "Input file(s)"};

     // 3) Parse with optional config file and unknown-args collection
     std::vector<std::string> unknown;
     auto pr = parse_with_config(args, "einsums", "2.0.0",
                                 "settings.json", &unknown);
     if (!pr.ok) return pr.exit_code;

     // 4) Use the values
     fmt::print("threads = {}\n", Threads.get());
     fmt::print("output  = {}\n", Output.get());
     fmt::print("verbose = {}\n", Verbose.get());
     for (auto& f : Inputs.values()) fmt::print("input   = {}\n", f);

     if (!unknown.empty()) {
       fmt::print("unknown: ");
       for (auto& u : unknown) fmt::print("{} ", u);
       fmt::print("\n");
     }
     return 0;
   }

Run it:

.. code-block:: console

   $ ./einsums -v -t 8 -o result.bin a.txt b.txt
   threads = 8
   output  = result.bin
   verbose = true
   input   = a.txt
   input   = b.txt

Built‑ins:

.. code-block:: console

   $ ./einsums --help
   Usage: einsums [options] <inputs>
   ...

   $ ./einsums --version
   einsums 2.0.0

Core Concepts
-------------

Categories
~~~~~~~~~~

``OptionCategory`` groups options under a heading in ``--help``.

.. code-block:: cpp

   OptionCategory IO{"I/O"};
   OptionCategory Perf{"Performance"};

Types of Options
~~~~~~~~~~~~~~~~

- ``Flag`` — boolean option. Presence sets true (configurable via ``ImplicitValue``).
- ``Opt<T>`` — single value option. ``T`` can be ``int``, ``double``, ``std::string``, …
- ``List<T>`` — repeated or comma‑separated values. As a positional, it *gathers remaining tokens*.
- ``OptEnum<Enum>`` — map string choices to an enum.
- ``Alias`` — forwards to a target option, optionally supplying a preset value.

Named Parameter Tags
~~~~~~~~~~~~~~~~~~~~

- ``Default(value)`` — compile‑time default.
- ``ImplicitValue(value)`` — used when the option appears without an explicit value.
- ``ValueName("NAME")`` — placeholder in help, e.g., ``--threads <N>``.

Bindings & Callbacks
~~~~~~~~~~~~~~~~~~~~

- ``Location<T>(ref)`` — write the parsed value directly into external storage.
- ``Setter<T>{ [](const T& v){ ... } }`` — invoke on assignment.

Occurrences & Visibility
~~~~~~~~~~~~~~~~~~~~~~~~

- ``Occurrence``: ``Optional`` (default), ``Required``, ``ZeroOrMore``, ``OneOrMore``.
- ``Visibility``: ``Normal`` (default), ``Hidden`` (omits from help).

API Reference
-------------

Construction (variadic, named‑style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Flag
   Flag(StringRef longName,
        std::initializer_list<char> shorts,
        StringRef helpText,
        /* extras: */ OptionCategory&, Visibility, Occurrence,
                     Location<bool>, Setter<bool>, Default(bool), ImplicitValue(bool));

   // Opt<T> (without positional default)
   template <typename T>
   Opt(StringRef longName,
       std::initializer_list<char> shorts,
       StringRef helpText,
       /* extras: */ OptionCategory&, Visibility, Occurrence, ValueExpected,
                    Range, Location<T>, Setter<T>, Default(T),
                    ImplicitValue(T), ValueNameTag);

   // Opt<T> (with positional default value)
   template <typename T>
   Opt(StringRef longName,
       std::initializer_list<char> shorts,
       T defaultValue,
       StringRef helpText,
       /* extras: ... as above ... */);

   // List<T> (named)
   template <typename T>
   List(StringRef longName,
        std::initializer_list<char> shorts,
        StringRef helpText,
        /* extras: OptionCategory&, Visibility, Occurrence */);

   // List<T> (positional gather)
   template <typename T>
   List(StringRef positionalName, Positional, StringRef helpText);

   // OptEnum<Enum>
   template <typename Enum>
   OptEnum(StringRef longName,
           std::initializer_list<char> shorts,
           Enum defaultValue,
           std::initializer_list<std::pair<std::string, Enum>> mapping,
           StringRef helpText,
           /* extras: OptionCategory&, ... */);

   // Alias
   Alias(StringRef longName,
         std::initializer_list<char> shorts,
         OptionBase& target,
         StringRef helpText,
         /* extras: OptionCategory&, Visibility, Occurrence, std::string presetValue */);

Parsing
~~~~~~~

.. code-block:: cpp

   struct ParseResult { bool ok; int exit_code; };

   ParseResult parse(const std::vector<std::string>& args,
                     const char* programName = nullptr,
                     std::string_view version = {},
                     std::map<std::string, std::string, std::less<>>* config = nullptr,
                     std::vector<std::string>* unknown_args = nullptr);

   ParseResult parse_with_config(const std::vector<std::string>& args,
                                 const char* programName = nullptr,
                                 std::string_view version = {},
                                 std::string_view config_path = {},
                                 std::vector<std::string>* unknown_args = nullptr);

Behavior & Semantics
--------------------

Option Names
~~~~~~~~~~~~

- Long: ``--threads``, optionally ``--threads=8``.
- Short (bundles allowed): ``-v``, ``-abc``, ``-o12`` or ``-o 12``.

Implicit Values
~~~~~~~~~~~~~~~

For options with ``ValueExpected::ValueRequired``, the parser **only consumes the next token
as a value if it does not look like another option**. Otherwise it passes ``std::nullopt`` to the
option, allowing ``ImplicitValue(...)`` to apply. Examples:

- ``--level``  ⇒ if ``ImplicitValue(7)`` set, then 7
- ``--level=9`` ⇒ 9
- ``-l`` (last in bundle) ⇒ apply implicit value if configured

Numeric look‑ahead rule
~~~~~~~~~~~~~~~~~~~~~~~

Tokens like ``-5`` or ``-3.14`` are treated as **values** (not options) when they are expected
to be consumed as the next value.

Positional List Gathering
~~~~~~~~~~~~~~~~~~~~~~~~~

A positional ``List<T>`` stays "active" and gathers subsequent bare tokens. For example:

.. code-block:: cpp

   List<std::string> Inputs{"inputs", Positional{}, "Input files"};
   // ./app a.txt b.txt c.txt  ⇒ Inputs.values() == {"a.txt","b.txt","c.txt"}

Unknown Arguments
~~~~~~~~~~~~~~~~~

- Any unrecognized option (e.g., ``--weird``) or short (e.g., ``-z``) is appended to ``unknown_args``.
- Everything **after** a literal ``--`` is appended to ``unknown_args``.

Config Files
~~~~~~~~~~~~

``parse_with_config`` accepts:

- **Key/Value** (``.env``‑ish): lines of ``key = value``; ``#`` starts a comment.
- **Flat JSON object**: ``{"threads": 12, "output": "a.bin", "verbose": true}``.

Keys are matched to long option names (case‑insensitive). Precedence is:

``Default < Config < CLI``

Built‑in Options
~~~~~~~~~~~~~~~~

- ``--help`` (alias ``-h``)
- ``--version``

These are always present. The parser **short‑circuits**:

- On ``--help``: prints help and returns ``{ok=false, exit_code=0}``
- On ``--version``: prints version and returns ``{ok=false, exit_code=0}``

Help Layout
~~~~~~~~~~~

``--help`` prints usage, categories, normal‑visibility options, and positional arguments.
``ValueName("NAME")`` customizes the placeholder.

Validation & Errors
~~~~~~~~~~~~~~~~~~~

- Missing required options or invalid values cause ``{ok=false, exit_code=1}`` and a message to ``stderr``.
- ``RangeBetween(min, max)`` checks integral bounds on numeric ``Opt<T>``.
- ``OptEnum`` reports invalid choices and lists allowed keys.

Examples
--------

Flags
~~~~~

.. code-block:: cpp

   OptionCategory Log{"Logging"};
   Flag Verbose{
     "verbose", {'v'}, "Enable verbose logging",
     Log, Default(false), ImplicitValue(true)
   };

Integers with Range and Implicit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   OptionCategory Perf{"Performance"};
   Opt<int> Threads{
     "threads", {'t'}, Default(4), "Number of threads",
     Perf, RangeBetween(1, 256), ValueName("N")
   }.Implicit(16); // --threads ⇒ 16, --threads=8 ⇒ 8

Strings & Paths
~~~~~~~~~~~~~~~

.. code-block:: cpp

   OptionCategory IO{"I/O"};
   Opt<std::string> Output{
     "output", {'o'}, Default(std::string{"a.out"}), "Output file",
     IO, ValueName("PATH")
   };

Lists (named and positional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   List<std::string> Include{
     "include", {'I'}, "Include directories",
     IO
   }; // --include=inc1,inc2 or --include inc1 --include inc2

   List<std::string> Inputs{"inputs", Positional{}, "Input files"};
   // gathers bare tokens at end: ./app a b c

Enums
~~~~~

.. code-block:: cpp

   enum struct Mode { Fast, Accurate, Debug };
   OptionCategory Modes{"Mode"};

   OptEnum<Mode> ModeOpt{
     "mode", {}, Mode::Fast,
     { {"fast", Mode::Fast}, {"accurate", Mode::Accurate}, {"debug", Mode::Debug} },
     "Execution mode", Modes
   };

Aliases
~~~~~~~

.. code-block:: cpp

   // Make --fast an alias for --mode=fast
   Alias FastAlias{
     "fast", {}, ModeOpt, "Alias: fast mode", Modes, std::string("fast")
   };

Binding and Callbacks
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   OptionCategory Tuning{"Tuning"};

   // Bind to external storage
   int threads_bound = 0;
   Opt<int> Threads{
     "threads", {'t'}, Default(4), "Threads",
     Tuning, Location<int>(threads_bound)
   };

   // Callback on set (config or CLI). from_config distinguishes source.
   Opt<int> Batch{
     "batch", {}, Default(32), "Batch size",
     Tuning, Setter<int>{[](int v, bool from_cfg){
       fmt::print("Reconfiguring batch={} (source: {})\n",
                  v, from_cfg ? "config" : "cli");
     }}
   };

Thread‑Safety & Reentrancy
--------------------------

- The design uses a **global registry** and **mutable option state** → **not thread‑safe** and
  **not reentrant** for concurrent parses. Serialize calls to ``parse(...)`` if needed.
- In unit tests, keeping options/categories **local** and clearing the registry per case/section
  provides deterministic isolation.
- Optional (coarse) hardening for apps that might parse concurrently:

  .. code-block:: cpp

     // Guarded parse wrapper sketch
     #include <mutex>
     std::mutex& cli_parse_mutex(){ static std::mutex m; return m; }
     // Acquire lock before calling parse_internal(...)

FAQ
---

How do implicit values work with short options?
  If a short option that requires a value is the last in a bundle (``-l``), the parser will consume
  the next token only if it **doesn’t look like an option**; otherwise ``ImplicitValue(...)`` applies.

Why are unknown options not an error?
  To emulate LLVM’s flexibility and to ease integration with upstream tools, unknown tokens and
  passthrough arguments (after ``--``) are returned to the caller in ``unknown_args`` for further dispatch.

Can I bind directly into application config structs?
  Yes—use ``Location<T>`` to a field that outlives parsing, or use ``Setter<T>`` to translate/
  validate and write into your own structure.

See the :ref:`API reference <modules_Einsums_CommandLine_api>` of this module for more
details.