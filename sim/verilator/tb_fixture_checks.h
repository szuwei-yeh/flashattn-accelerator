#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Hex files encode unsigned bit patterns, not signed text. Reject overflow,
// partial tokens and extra entries instead of silently truncating the fixture.
template <typename T>
inline bool load_hex_vector(const std::string& path, std::vector<T>& out, int count) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 4, "Expected byte or INT32 fixture");
    std::ifstream input(path);
    std::string token;
    out.clear();
    if (!input || count <= 0) {
        std::fprintf(stderr, "ERROR: cannot read vector %s\n", path.c_str());
        return false;
    }
    while (input >> token) {
        if (out.size() >= static_cast<size_t>(count) || token.size() > sizeof(T) * 2 ||
            token.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
            std::fprintf(stderr, "ERROR: invalid hex vector %s at entry %zu\n",
                         path.c_str(), out.size());
            return false;
        }
        const uint64_t bits = std::stoull(token, nullptr, 16);
        const uint64_t modulus = uint64_t{1} << (sizeof(T) * 8);
        const int64_t value = bits >= modulus / 2 ?
            static_cast<int64_t>(bits) - static_cast<int64_t>(modulus) :
            static_cast<int64_t>(bits);
        out.push_back(static_cast<T>(value));
    }
    if (input.bad() || out.size() != static_cast<size_t>(count)) {
        std::fprintf(stderr, "ERROR: wrong hex vector length in %s: got %zu, expected %d\n",
                     path.c_str(), out.size(), count);
        return false;
    }
    return true;
}

inline bool check_tb_geometry(int n, int d, int built_n, int built_d) {
    if (n != built_n || d != built_d || n <= 0 || d <= 0) {
        std::fprintf(stderr, "ERROR: testbench geometry N=%d D=%d differs from compiled N=%d D=%d\n",
                     n, d, built_n, built_d);
        return false;
    }
    return true;
}

// Scale files are required inputs, including for all-zero output fixtures.
// Metadata such as N/d/seed is allowed; all three scale keys must occur once.
inline bool load_required_scales(const std::string& path, uint16_t& sq,
                                 uint16_t& sk, uint16_t& sv) {
    std::ifstream input(path);
    unsigned seen = 0;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key, equals, token, extra;
        row >> key;
        const unsigned bit = key == "scale_q_q88" ? 1 :
                             key == "scale_k_q88" ? 2 :
                             key == "scale_v_q88" ? 4 : 0;
        if (!bit) continue;
        if ((seen & bit) || !(row >> equals >> token) || equals != "=" ||
            (row >> extra) || token.size() < 3 || token.size() > 6 ||
            token.substr(0, 2) != "0x" ||
            token.find_first_not_of("0123456789abcdefABCDEF", 2) != std::string::npos) {
            std::fprintf(stderr, "ERROR: invalid scale entry in %s\n", path.c_str());
            return false;
        }
        const auto value = static_cast<uint16_t>(std::stoul(token, nullptr, 16));
        if (bit == 1) sq = value;
        if (bit == 2) sk = value;
        if (bit == 4) sv = value;
        seen |= bit;
    }
    if (seen != 7 || input.bad()) {
        std::fprintf(stderr, "ERROR: missing required scales in %s\n", path.c_str());
        return false;
    }
    return true;
}
