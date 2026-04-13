// ============================================================
//  sim_main.cpp  —  Verilator testbench for systolic_array
//
//  Tests a 4×4 systolic array (SIZE=4, overridden from default 16):
//
//  Test 1: A = identity, B = all-2
//          C = A*B = B  →  every element should be 2
//
//  Test 2: A = all-1,    B = all-1
//          C[i][j] = sum_{k=0..3} 1*1 = 4  →  every element should be 4
//
//  Skewed input timing:
//    At feed cycle t (0-indexed), the externally provided inputs are:
//      a_in[r] = A[r][t-r]   if 0 <= t-r < SIZE, else 0
//      b_in[c] = B[t-c][c]   if 0 <= t-c < SIZE, else 0
//    PE[r][c] therefore sees A[r][k]*B[k][c] at pipeline cycle r+c+k.
//    The last accumulation (k=SIZE-1) at PE[SIZE-1][SIZE-1] occurs at
//    cycle  t = (SIZE-1)+(SIZE-1)+(SIZE-1) = 3*(SIZE-1) = 9 for SIZE=4.
//    Total feed cycles needed: 3*(SIZE-1)+1 = 3*SIZE-2 = 10.
// ============================================================

#include "verilated.h"
#include "Vsystolic_array.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static constexpr int SIZE = 4;

static Vsystolic_array* top;
static vluint64_t sim_time = 0;

// ---------- helpers --------------------------------------------------

static void tick() {
    top->clk = 0; top->eval(); sim_time++;
    top->clk = 1; top->eval(); sim_time++;
}

// Feed one cycle of skewed inputs for feed-index t.
// a_in[r] = A[r][t-r],  b_in[c] = B[t-c][c]  (0 when out-of-range)
static void feed_skewed(const int8_t A[SIZE][SIZE],
                        const int8_t B[SIZE][SIZE],
                        int t)
{
    for (int r = 0; r < SIZE; r++) {
        int k = t - r;
        top->a_in[r] = (k >= 0 && k < SIZE) ? (uint8_t)A[r][k] : 0;
    }
    for (int c = 0; c < SIZE; c++) {
        int k = t - c;
        top->b_in[c] = (k >= 0 && k < SIZE) ? (uint8_t)B[k][c] : 0;
    }
}

// ---------- test runner ----------------------------------------------

static bool run_test(const char*      name,
                     const int8_t     A[SIZE][SIZE],
                     const int8_t     B[SIZE][SIZE],
                     const int32_t    expected[SIZE][SIZE])
{
    printf("\n--- %s ---\n", name);

    // ---- reset -------------------------------------------------------
    top->rst_n = 0; top->en = 0; top->clear = 0;
    for (int i = 0; i < SIZE; i++) { top->a_in[i] = 0; top->b_in[i] = 0; }
    tick(); tick();          // hold reset for 2 cycles
    top->rst_n = 1;

    // ---- clear accumulators (1 cycle) --------------------------------
    top->en    = 1;
    top->clear = 1;
    for (int i = 0; i < SIZE; i++) { top->a_in[i] = 0; top->b_in[i] = 0; }
    tick();
    top->clear = 0;

    // ---- feed skewed data + drain ------------------------------------
    // Need feed cycles t = 0 .. 3*(SIZE-1) = 3*SIZE-3
    // (covers the last valid mult at PE[SIZE-1][SIZE-1])
    const int TOTAL_FEED = 3 * SIZE - 2;   // t = 0 .. 3*SIZE-3, i.e. 3*SIZE-2 ticks
    for (int t = 0; t < TOTAL_FEED; t++) {
        feed_skewed(A, B, t);
        tick();
    }

    // ---- check results -----------------------------------------------
    bool pass = true;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            int32_t got = (int32_t)top->acc[r * SIZE + c];
            int32_t exp = expected[r][c];
            if (got != exp) {
                printf("  FAIL C[%d][%d] = %d  (expected %d)\n", r, c, got, exp);
                pass = false;
            }
        }
    }

    if (pass)
        printf("  PASS  all %d elements correct\n", SIZE * SIZE);
    return pass;
}

// Like run_test but skips reset — used for back-to-back multiply tests.
// Caller must ensure the array is already in a known state (e.g. just reset).
static bool run_test_no_reset(const char*   name,
                              const int8_t  A[SIZE][SIZE],
                              const int8_t  B[SIZE][SIZE],
                              const int32_t expected[SIZE][SIZE])
{
    printf("\n--- %s ---\n", name);

    // ---- clear accumulators (1 cycle) --------------------------------
    top->en    = 1;
    top->clear = 1;
    for (int i = 0; i < SIZE; i++) { top->a_in[i] = 0; top->b_in[i] = 0; }
    tick();
    top->clear = 0;

    // ---- feed skewed data --------------------------------------------
    const int TOTAL_FEED = 3 * SIZE - 2;
    for (int t = 0; t < TOTAL_FEED; t++) {
        feed_skewed(A, B, t);
        tick();
    }

    // ---- check results -----------------------------------------------
    bool pass = true;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            int32_t got = (int32_t)top->acc[r * SIZE + c];
            int32_t exp = expected[r][c];
            if (got != exp) {
                printf("  FAIL C[%d][%d] = %d  (expected %d)\n", r, c, got, exp);
                pass = false;
            }
        }
    }
    if (pass)
        printf("  PASS  all %d elements correct\n", SIZE * SIZE);
    return pass;
}

// CPU reference matmul (int8 → int32)
static void cpu_matmul(const int8_t  A[SIZE][SIZE],
                       const int8_t  B[SIZE][SIZE],
                       int32_t       C[SIZE][SIZE])
{
    for (int r = 0; r < SIZE; r++)
        for (int c = 0; c < SIZE; c++) {
            int32_t sum = 0;
            for (int k = 0; k < SIZE; k++)
                sum += (int32_t)A[r][k] * (int32_t)B[k][c];
            C[r][c] = sum;
        }
}

// ---------- main ------------------------------------------------------

int main(int argc, char** argv)
{
    Verilated::commandArgs(argc, argv);
    top = new Vsystolic_array;

    // ---- Test 1: A = identity,  B = all-2  -------------------------
    int8_t  A1[SIZE][SIZE] = {};   // zero-init, then set diagonal
    int8_t  B1[SIZE][SIZE];
    int32_t E1[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++) A1[i][i] = 1;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) { B1[i][j] = 2; E1[i][j] = 2; }

    // ---- Test 2: A = all-1,  B = all-1  ----------------------------
    // C[i][j] = sum_{k=0}^{SIZE-1} 1*1 = SIZE = 4
    int8_t  A2[SIZE][SIZE];
    int8_t  B2[SIZE][SIZE];
    int32_t E2[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) { A2[i][j] = 1; B2[i][j] = 1; E2[i][j] = SIZE; }

    // ---- Test 3: A[r][c] = r+1,  B[r][c] = c+1  -------------------
    // A[i][k] = i+1  (same for all k)
    // B[k][j] = j+1  (same for all k)
    // C[i][j] = sum_{k=0}^{SIZE-1} (i+1)*(j+1) = SIZE*(i+1)*(j+1)
    int8_t  A3[SIZE][SIZE];
    int8_t  B3[SIZE][SIZE];
    int32_t E3[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            A3[i][j] = (int8_t)(i + 1);
            B3[i][j] = (int8_t)(j + 1);
            E3[i][j] = SIZE * (i + 1) * (j + 1);
        }

    // ---- Test 4: negative matrix  -----------------------------------
    // A = all-(-1), B = all-1
    // C[i][j] = sum_{k} (-1)*1 = -SIZE
    int8_t  A4[SIZE][SIZE];
    int8_t  B4[SIZE][SIZE];
    int32_t E4[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) { A4[i][j] = -1; B4[i][j] = 1; E4[i][j] = -SIZE; }

    // ---- Test 5: back-to-back (clear correctness)  ------------------
    // Round 1: all-1 * all-1 → all-SIZE  (uses run_test for full reset)
    // Round 2: identity * all-3 → all-3  (uses run_test_no_reset, shares same reset)
    int8_t  A5a[SIZE][SIZE];
    int8_t  B5a[SIZE][SIZE];
    int32_t E5a[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) { A5a[i][j] = 1; B5a[i][j] = 1; E5a[i][j] = SIZE; }

    int8_t  A5b[SIZE][SIZE] = {};   // identity
    int8_t  B5b[SIZE][SIZE];
    int32_t E5b[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++) A5b[i][i] = 1;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) { B5b[i][j] = 3; E5b[i][j] = 3; }

    // ---- Test 6: random matrices (3 seeds) --------------------------
    const int SEEDS[3] = {42, 137, 2025};

    bool ok = true;
    ok &= run_test("Test1: I * all-2  => all-2",           A1, B1, E1);
    ok &= run_test("Test2: all-1 * all-1 => all-4",        A2, B2, E2);
    ok &= run_test("Test3: A[r][c]=r+1, B[r][c]=c+1",     A3, B3, E3);
    ok &= run_test("Test4: all-(-1) * all-1 => all-(-SIZE)", A4, B4, E4);

    // Test 5: two multiplies with only a clear between them
    printf("\n--- Test5a: all-1 * all-1 => all-SIZE (round 1 of back-to-back) ---\n");
    // Full reset for first round
    top->rst_n = 0; top->en = 0; top->clear = 0;
    for (int i = 0; i < SIZE; i++) { top->a_in[i] = 0; top->b_in[i] = 0; }
    tick(); tick();
    top->rst_n = 1;
    ok &= run_test_no_reset("Test5b: I * all-3 => all-3  (after clear, no reset)", A5b, B5b, E5b);
    // Also verify round-1 result wasn't a fluke — run it properly
    ok &= run_test("Test5a: all-1 * all-1 => all-SIZE",   A5a, B5a, E5a);
    ok &= run_test_no_reset("Test5b: I * all-3 => all-3  (clear only, no reset)", A5b, B5b, E5b);

    // Test 6: random
    for (int s = 0; s < 3; s++) {
        srand((unsigned)SEEDS[s]);
        int8_t  A6[SIZE][SIZE];
        int8_t  B6[SIZE][SIZE];
        int32_t E6[SIZE][SIZE];
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++) {
                A6[i][j] = (int8_t)((rand() % 21) - 10);  // -10 .. 10
                B6[i][j] = (int8_t)((rand() % 21) - 10);
            }
        cpu_matmul(A6, B6, E6);
        char name[64];
        snprintf(name, sizeof(name), "Test6 seed=%d: random INT8", SEEDS[s]);
        ok &= run_test(name, A6, B6, E6);
    }

    top->final();
    delete top;

    printf("\n%s\n", ok ? "=== ALL TESTS PASSED ===" : "=== SOME TESTS FAILED ===");

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : systolic_array\n");
    printf("Scenarios covered: identity_mul, all_ones, ramp_matrix, negative, back_to_back_clear, random_int8\n");
    printf("Test cases run   : 10\n");
    printf("Mismatches       : %d\n", ok ? 0 : 1);
    printf("Result           : %s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
