//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/InitModule.hpp>
#include <Einsums/Tensor/ModuleVars.hpp>

#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <mutex>
#include <string>

namespace einsums {

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 *
 * Logging will not be available by the time the initialization routines are run.
 */

int setup_Einsums_Tensor() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    // If you don't need a function, you may remove its respective line from the
    // if statement below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_arguments(einsums::add_Einsums_Tensor_arguments);
        einsums::register_startup_function(einsums::initialize_Einsums_Tensor);
        einsums::register_shutdown_function(einsums::finalize_Einsums_Tensor);

        is_initialized = true;
    }

    return 0;
}

EINSUMS_EXPORT void add_Einsums_Tensor_arguments(argparse::ArgumentParser &parser) {
    /** @todo Fill in.
     *
     * If you are not using one of the following maps, you may remove references to it.
     * The maps may not need to be locked before use, but it should still be done just in
     * case.
     */

    auto pid = getpid();

    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_string = global_config.get_string_map()->get_value();
    auto &global_double = global_config.get_double_map()->get_value();
    auto &global_int    = global_config.get_int_map()->get_value();
    auto &global_bool   = global_config.get_bool_map()->get_value();

    auto lock = std::lock_guard(global_config);

    parser.add_argument("--einsums:scratch-dir")
        .default_value(std::filesystem::temp_directory_path().string())
        .help("The scratch directory for Einsums tensor files.")
        .store_into(global_string["scratch-dir"]);

    parser.add_argument("--einsums:hdf5-file-name")
        .default_value(fmt::format("einsums.{}.h5", pid))
        .help("The name of the HDF5 file for Einsums. Defaults to einsums.[pid].h5, where [pid] is the PID of the current process.")
        .store_into(global_string["hdf5-file-name"]);

    parser.add_argument("--einsums:no-delete-hdf5-files")
        .default_value(true)
        .implicit_value(false)
        .help("Tells Einsums to clean up HDF5 files on exit.")
        .store_into(global_bool["delete-hdf5-files"]);

    /* Examples:
     * String argument:
     * parser.add_argument("--einsums:dummy-str")
     *   .default_value("default value")
     *   .help("Dummy string argument.")
     *   .store_into(global_string["dummy-str"]);
     *
     * Double argument:
     * parser.add_argument("--einsums:dummy-double")
     *   .default_value(1.23)
     *   .help("Dummy double argument.")
     *   .store_into(global_double["dummy-double"]);
     *
     * Integer argument: Be careful! These are int64_t, not int, so
     * literals need to be handled with care.
     *
     * parser.add_argument("--einsums:dummy-int")
     *   .default_value<int64_t>(1)
     *   .help("Dummy int argument.")
     *   .store_into(global_int["dummy-int"]);
     *
     * parser.add_argument("--einsums:dummy-int2")
     *   .default_value(1L)
     *   .help("Dummy int argument, another way.")
     *   .store_into(global_int["dummy-int2"]);
     *
     * Boolean argument:
     * This one adds a flag that is normally false, but is set when provided.
     * parser.add_argument("--einsums:dummy-bool")
     *   .flag()
     *   .help("Dummy int argument.")
     *   .store_into(global_int["dummy-bool"]);
     *
     * This one adds a flag that is normally true, but is unset when provided.
     * parser.add_argument("--einsums:no-dummy-bool2")
     *   .default_value(true)
     *   .implicit_value(false)
     *   .help("Dummy int argument.")
     *   .store_into(global_int["dummy-bool2"]);
     */
}

void initialize_Einsums_Tensor() {
    auto &singleton     = einsums::detail::Einsums_Tensor_vars::get_singleton();
    auto &global_config = GlobalConfigMap::get_singleton();

    auto fname = std::filesystem::path(global_config.get_string("scratch-dir"));
    fname /= global_config.get_string("hdf5-file-name");

    singleton.hdf5_file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    if (singleton.hdf5_file == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("HDF5 file could not be opened!");
        std::terminate();
    }

    singleton.link_property_list = H5Pcreate(H5P_LINK_CREATE);

    if (singleton.link_property_list == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("Could not create HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
    }

    int res = H5Pset_char_encoding(singleton.link_property_list, H5T_CSET_UTF8);
    res *= H5Pset_create_intermediate_group(singleton.link_property_list, 1);

    if (res < 0) {
        EINSUMS_LOG_ERROR("Could not apply properties to the HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        std::terminate();
    }

    singleton.double_complex_type = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
    singleton.float_complex_type  = H5Tcreate(H5T_COMPOUND, 2 * sizeof(float));

    if (singleton.double_complex_type == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("Could not create HDF5 double complex number data type!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        std::terminate();
    } else {
        int err = 1;
        err *= H5Tinsert(singleton.double_complex_type, "x", 0, H5T_NATIVE_DOUBLE);
        err *= H5Tinsert(singleton.double_complex_type, "y", 1, H5T_NATIVE_DOUBLE);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not assign members to double complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            std::terminate();
        }
    }

    if (singleton.float_complex_type == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("Could not create HDF5 float complex number data type!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        H5Fclose(singleton.double_complex_type);
        std::terminate();
    } else {
        int err = 1;
        err *= H5Tinsert(singleton.float_complex_type, "x", 0, H5T_NATIVE_FLOAT);
        err *= H5Tinsert(singleton.float_complex_type, "y", 1, H5T_NATIVE_FLOAT);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not assign members to FLOAT complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            H5Fclose(singleton.float_complex_type);
            std::terminate();
        }
    }
}

void finalize_Einsums_Tensor() {
    auto &singleton     = einsums::detail::Einsums_Tensor_vars::get_singleton();
    auto &global_config = GlobalConfigMap::get_singleton();

    auto fname = std::filesystem::path(global_config.get_string("scratch-dir"));
    fname /= global_config.get_string("hdf5-file-name");

    H5Fclose(singleton.hdf5_file);

    if (singleton.hdf5_file != H5I_INVALID_HID && global_config.get_bool("delete-hdf5-files")) {
        H5Fdelete(fname.c_str(), H5P_DEFAULT);
    }

    if (singleton.double_complex_type != H5I_INVALID_HID) {
        H5Tclose(singleton.double_complex_type);
    }

    if (singleton.float_complex_type != H5I_INVALID_HID) {
        H5Tclose(singleton.float_complex_type);
    }
}

} // namespace einsums