#include <cstdint>
#include <cstdio>
#include <limits>

#include "Vtb_dequantizer_combined.h"
#include "verilated.h"

static void tick(Vtb_dequantizer_combined* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

static int64_t arithmetic_shift_right_8(int64_t value) {
    if (value >= 0)
        return value / 256;
    return -static_cast<int64_t>((static_cast<uint64_t>(-(value + 1)) + 256) / 256);
}

static int16_t reference_dequantize(int32_t data, int16_t scale_q, int16_t scale_k) {
    const int64_t mid     = static_cast<int64_t>(data) * scale_q;
    const int64_t product = mid * scale_k;
    const int64_t shifted = arithmetic_shift_right_8(product + 128);

    if (shifted > std::numeric_limits<int16_t>::max())
        return std::numeric_limits<int16_t>::max();
    if (shifted < std::numeric_limits<int16_t>::min())
        return std::numeric_limits<int16_t>::min();
    return static_cast<int16_t>(shifted);
}

struct TestCase {
    const char* name;
    int32_t     data;
    int16_t     scale_q;
    int16_t     scale_k;
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vtb_dequantizer_combined;

    dut->rst_n = 0;
    dut->valid_in = 0;
    dut->data_in = 0;
    dut->scale_q = 0;
    dut->scale_k = 0;
    tick(dut);
    tick(dut);
    dut->rst_n = 1;
    tick(dut);

    const TestCase tests[] = {
        {"zero",                         0,             6,      7},
        {"positive_data",                12345,         6,      7},
        {"negative_data",               -12345,         6,      7},
        {"negative_q_scale",               123,      -300,    200},
        {"negative_k_scale",               123,       300,   -200},
        {"two_negative_scales",            123,      -300,   -200},
        {"positive_round_up",                1,         1,    129},
        {"negative_rounding_behavior",       -1,         1,    129},
        {"positive_saturation_d16_bound", 262144,       256,    256},
        {"negative_saturation_d16_bound",-260096,       256,    256},
        {"positive_saturation_d64_bound",1048576,       256,    256},
        {"negative_saturation_d64_bound",-1040384,      256,    256},
        {"combined_scale_positive_corner",    1,    -32768, -32768},
        {"combined_scale_negative_corner",    1,    -32768,  32767},
        {"max_data_mixed_sign",  std::numeric_limits<int32_t>::max(), 1, -1},
        {"min_data_mixed_sign",  std::numeric_limits<int32_t>::min(), 1, -1},
    };

    int failures = 0;
    for (const auto& test : tests) {
        const int64_t old_mid = static_cast<int64_t>(test.data) * test.scale_q;
        const int64_t old_product = old_mid * test.scale_k;
        const int32_t combined = static_cast<int32_t>(
            static_cast<int64_t>(test.scale_q) * test.scale_k);
        const int64_t combined_product = static_cast<int64_t>(test.data) * combined;

        if (old_product != combined_product) {
            std::printf("FAIL %-36s reassociation mismatch old=%lld new=%lld\n",
                        test.name, static_cast<long long>(old_product),
                        static_cast<long long>(combined_product));
            failures++;
            continue;
        }

        dut->data_in = static_cast<uint32_t>(test.data);
        dut->scale_q = static_cast<uint16_t>(test.scale_q);
        dut->scale_k = static_cast<uint16_t>(test.scale_k);
        dut->valid_in = 1;
        tick(dut);

        const int16_t expected = reference_dequantize(
            test.data, test.scale_q, test.scale_k);
        const int16_t actual = static_cast<int16_t>(dut->data_out);
        const int32_t actual_combined = static_cast<int32_t>(dut->dbg_combined_scale);

        if (!dut->valid_out || actual != expected || actual_combined != combined) {
            std::printf("FAIL %-36s valid=%u output=%d expected=%d combined=%d expected_combined=%d\n",
                        test.name, dut->valid_out, actual, expected,
                        actual_combined, combined);
            failures++;
        } else {
            std::printf("PASS %-36s output=%d combined=%d\n",
                        test.name, actual, actual_combined);
        }

        dut->valid_in = 0;
        tick(dut);
        if (dut->valid_out) {
            std::printf("FAIL %-36s valid_out did not deassert\n", test.name);
            failures++;
        }
    }

    delete dut;
    if (failures != 0) {
        std::printf("\nRESULT: FAIL (%d failures)\n", failures);
        return 1;
    }

    std::printf("\nRESULT: PASS (%zu bit-exact signed/rounding/saturation cases)\n",
                sizeof(tests) / sizeof(tests[0]));
    return 0;
}
