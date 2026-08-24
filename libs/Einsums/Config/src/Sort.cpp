//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/StdAlternatives.hpp>

#include <algorithm>

// Apparently we have to do this ourselves. It doesn't seem to be possible to convert a lambda with captures to an
// appropriate function pointer. At least not in a way that is portable across compilers.

static void insertion_sort(void *base, std::size_t elements, std::size_t width, einsums::safe_compare compare, void *context) {
    char *char_base = reinterpret_cast<char *>(base);

    for (std::size_t head = 0; head < elements - 1; head++) {
        char *head_ptr = char_base + head * width;

        char *min_ptr = head_ptr;

        for (std::size_t check = head + 1; check < elements; check++) {
            char *curr_ptr = char_base + check * width;
            if (compare(context, reinterpret_cast<void *>(curr_ptr), reinterpret_cast<void *>(min_ptr)) < 0) {
                min_ptr = curr_ptr;
            }
        }

        if (min_ptr != head_ptr) {
            std::swap_ranges(head_ptr, head_ptr + width, min_ptr);
        }
    }
}

static std::size_t partition_elements(void *base, std::size_t elements, std::size_t width, einsums::safe_compare compare, void *context) {
    char *char_base = reinterpret_cast<char *>(base);
    char *pivot     = char_base + width * (elements - 1);

    // Pick the final value as the pivot. Initial could work too, as could a random value.

    std::size_t out = 0;

    for (std::size_t i = 0; i < elements - 1; i++) {
        if (compare(context, char_base + i * width, pivot) <= 0) {
            if (i != out) {
                std::swap_ranges(char_base + i * width, char_base + (i + 1) * width, char_base + out * width);
            }
            out++;
        }
    }

    // Swap the pivot into the pivot position. Everything less than or equal to the pivot is before out,
    // so the value at out must be greater than the pivot.
    std::swap_ranges(char_base + out * width, char_base + (out + 1) * width, pivot);

    return out;
}

namespace einsums {

void qsort_s(void *base, std::size_t elements, std::size_t width, safe_compare compare, void *context) {
    // The size where insertion sort becomes faster.
    constexpr int insertion_size = 16;

    char *const char_base = reinterpret_cast<char *>(base);

    // We can't sort one element or zero elements.
    if (elements <= 1) {
        return;
    }

    // If we have a small case, non-recursive insertion sort is faster than recursive quicksort.
    if (elements <= insertion_size) {
        insertion_sort(base, elements, width, compare, context);
    } else {
        // Otherwise, do quicksort.

        std::size_t split_point = partition_elements(base, elements, width, compare, context);

        // Sort the bit before the split.
        qsort_s(base, split_point, width, compare, context);

        // Sort the bit after the split.
        if (split_point + 1 < elements) {
            qsort_s(reinterpret_cast<void *>(char_base + (split_point + 1) * width), elements - (split_point + 1), width, compare, context);
        }
    }
}
} // namespace einsums