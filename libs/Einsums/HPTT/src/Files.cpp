#include <Einsums/HPTT/Files.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace hptt {

constexpr char endian_char() {
    if constexpr (std::endian::native == std::endian::big) {
        return 'B';
    } else if constexpr (std::endian::native == std::endian::little) {
        return 'L';
    } else {
        throw std::runtime_error("Mixed endian systems are not supported.");
    }
}

template <std::integral T>
constexpr T byteswap(T value) noexcept {
    auto bytes = std::bit_cast<std::array<uint8_t, sizeof(T)>>(value);
    std::ranges::reverse(bytes);
    return std::bit_cast<T>(bytes);
}

template <>
constexpr uint8_t byteswap(uint8_t value) noexcept {
    return value;
}

template <>
constexpr uint16_t byteswap(uint16_t value) noexcept {
    union {
        uint16_t whole;
        uint8_t  bytes[2];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[1]);
    return convert.whole;
}

template <>
constexpr uint32_t byteswap(uint32_t value) noexcept {
    union {
        uint32_t whole;
        uint8_t  bytes[4];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[3]);
    std::swap(convert.bytes[1], convert.bytes[2]);
    return convert.whole;
}

template <>
constexpr int32_t byteswap(int32_t value) noexcept {
    union {
        int32_t whole;
        uint8_t bytes[4];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[3]);
    std::swap(convert.bytes[1], convert.bytes[2]);
    return convert.whole;
}

constexpr char version[4] = {0, 0, 0, endian_char()};

void setupFile(std::FILE *fp) {
    FileHeader header{.magic{'H', 'P', 'T', 'T'}, .version{version[0], version[1], version[2], version[3]}, .checksum = 0xffffffff};

    int error = std::fseek(fp, 0, SEEK_SET);

    if (error != 0) {
        std::perror("Error encountered while trying to rewind HPTT file!");
        throw std::runtime_error("IO error");
    }

    size_t err2 = std::fwrite(&header, sizeof(FileHeader), 1, fp);

    if (err2 < 1) {
        std::perror("Error encountered while writing to HPTT file!");
        throw std::runtime_error("IO error");
    }
}

uint32_t computeChecksum(std::FILE *fp) {
    int error = fseek(fp, 0, SEEK_END);
    if (error != 0) {
        std::perror("Error encountered while trying to seek in HPTT file!");
        throw std::runtime_error("IO error");
    }

    long end_pos = ftell(fp);

    error = fseek(fp, 7, SEEK_SET);

    if (error != 0) {
        perror("Error encountered while trying to seek in HPTT file!");
        throw std::runtime_error("IO error");
    }

    char file_endian_char = (char)fgetc(fp);

    if (file_endian_char != 'B' && file_endian_char != 'L') {
        throw std::runtime_error("Malformed HPTT file! Endianness specification is broken!");
    }

    error = fseek(fp, sizeof(FileHeader), SEEK_SET);

    if (error != 0) {
        perror("Error encountered while trying to seek in HPTT file!");
        throw std::runtime_error("IO error");
    }

    size_t data_len = end_pos - ftell(fp);
    data_len /= 2;

    // This uses Fletcher-32 from Wikipedia.

    uint32_t sum0 = 0, sum1 = 0;

    uint16_t data_buff[360];

    if (endian_char() != file_endian_char) {
        while (data_len > 0) {
            size_t block_len = data_len;
            if (block_len > 360) {
                block_len = 360;
            }
            data_len -= block_len;

            error = fread(data_buff, 2, block_len, fp);

            if (error < block_len) {
                perror("Error while reading file!");
                throw std::runtime_error("IO error");
            }

            for (int i = 0; i < block_len; i++) {
                sum0 += byteswap(data_buff[i]);
                sum1 += sum0;
            }

            sum0 %= 65535;
            sum1 %= 65535;
        }
    } else {
        while (data_len > 0) {
            size_t block_len = data_len;
            if (block_len > 360) {
                block_len = 360;
            }
            data_len -= block_len;

            error = fread(data_buff, 2, block_len, fp);

            if (error < block_len) {
                perror("Error while reading file!");
                throw std::runtime_error("IO error");
            }

            for (int i = 0; i < block_len; i++) {
                sum0 += data_buff[i];
                sum1 += sum0;
            }

            sum0 %= 65535;
            sum1 %= 65535;
        }
    }

    return (sum1 << 16 | sum0);
}

int verifyFile(std::FILE *fp) {
    FileHeader header;

    int error = fseek(fp, 0, SEEK_SET);

    if (error != 0) {
        return 1;
    }

    size_t read_bytes = fread(&header, sizeof(FileHeader), 1, fp);

    if (read_bytes < 1) {
        return 2;
    }

    // Check for the magic.
    if (strncmp(header.magic, "HPTT", 4) != 0) {
        return 3;
    }

    if (header.version[3] != endian_char()) {
        return 4;
    }

    uint32_t checksum = computeChecksum(fp);
    if (checksum != header.checksum) {
        return 5;
    }

    return 0;
}

} // namespace hptt