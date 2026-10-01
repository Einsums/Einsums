//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/ModuleVars.hpp>

namespace einsums::detail {

EINSUMS_SINGLETON_IMPL(Einsums_Tensor_vars)


std::string Einsums_Tensor_vars::get_temp_name() {
    constexpr static char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    static_assert(sizeof(base64_chars) >= 64);
    
    int64_t sequence = temp_counter.fetch_add(1);
    
    // Parameters from the Wikipedia article on the Fowler-Noll-Vo hash function. 
    constexpr int64_t fnv_prime = 0x00000100000001b3;
    constexpr int64_t fnv_offset = 0xcbf29ce484222325;
    
    int64_t hash = fnv_offset;
    
    for(int i = 0; i < 8; i++) {
        // Get the next byte of data.
        uint8_t byte = sequence & 0xff;
        sequence >>= 8;
        
        // Perform the FNV update.
        hash ^= byte;
        hash *= fnv_prime;
    }
    
    // Convert the hash to a base64 string.
    char base64_str[12];
    
    for(int i = 0; i < 11; i++) {
        uint8_t index = hash & 0x3f;
        hash >>= 6;
        
        base64_str[i] = base64_chars[index];
    }
    base64_str[11] = 0;
    
    // These names are for temporary tensors in an HDF5 file.
    return einsums::detail::corrected_format("tensor_{}", base64_str);
    
}

}