/*
 * Copyright (c) 2022 Justin Turney
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file Section.hpp
 *
 * Prints sections in the output file.
 */

#pragma once

#include "einsums/_Export.hpp"

#include <memory>
#include <string>

#include "einsums/_Export.hpp"

/**
 * @struct Section
 * 
 * Represents a unique section in the output file.
 */
struct EINSUMS_EXPORT Section {
    struct Impl;

    /**
     * Construct a new section.
     * 
     * @param name The name of the section.
     * @param pushTimer Whether to start a new timer for this section.
     */
    explicit Section(const std::string &name, bool pushTimer = true);

    /**
     * Construct a new section in the given domain.
     *
     * @param name The name of the section.
     * @param domain The domain of the section.
     * @param pushTimer Whether to start a new timer for this section.
     */
    Section(const std::string &name, const std::string &domain, bool pushTimer = true);
    
    /**
     * Deconstruct this section.
     */
    ~Section();

    /**
     * End this section.
     */
    void end();

  private:
    /**
     * Begin this section.
     */
    void begin();

    std::unique_ptr<Impl> _impl;
};

// Use of LabeledSection requires fmt/format.h to be included and the use of
// (BEGIN|END)_EINSUMS_NAMESPACE_(CPP|HPP)() defined in _Common.hpp

/**
 * @def LabeledSection1(x)
 *
 * Creates a new section with the given name.
 */
#define LabeledSection1(x) Section _section(fmt::format("{}::{} {}", detail::s_Namespace, __func__, x))

/**
 * @def LabeledSection0()
 *
 * Creates a new section with a default name.
 */
#define LabeledSection0()  Section _section(fmt::format("{}::{}", detail::s_Namespace, __func__))
