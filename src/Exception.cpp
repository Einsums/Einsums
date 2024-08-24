#include "einsums.hpp"

std::string einsums::detail::anonymize(std::string fpath) {

#ifdef EINSUMS_ANONYMIZE
    std::filesystem::path path(fpath);

    // Grab the file name.
    auto filename = path.filename();
    // Grab the parent path.
    auto parent = path.parent_path();

    std::filesystem::path out(filename);

    /*
     * Following parents will either lead to src, tests, timing, or none of these. If it does,
     * then stop there. If it doesn't, then it should reach include.
     * If it reaches src, then append it with the string /git.
     * If it doesn't, then append it with the string /install.
     */

    auto temp = parent;

    while (temp.has_filename()) {
        auto swap = temp.filename();
        swap /= out;
        out = swap;

        if (temp.filename() == "src" || temp.filename() == "tests" || temp.filename() == "timing") {
            swap = std::filesystem::path("/git");
            swap /= out;
            out = swap;
            return (std::string)out;
        }

        temp = temp.parent_path();
    }

    // If we got here, then there is no upper src directory. Go until include then.
    temp = parent;
    out  = filename;

    while (temp.has_filename()) {
        auto swap = temp.filename();
        swap /= out;
        out = swap;

        if (temp.filename() == "include") {
            swap = std::filesystem::path("/install");
            swap /= out;
            out = swap;
            return (std::string)out;
        }

        temp = temp.parent_path();
    }

    // If we got here, then there must be some issue when building. Return the file path without anonymizing.
    return fpath;

#else
    // If the user doesn't want to anonymize the paths, then just return the full raw path.
    return fpath;
#endif
}