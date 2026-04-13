// ============================================================
//  tb_array_controller.cpp  —  Verilator testbench for array_controller
//
//  Uses the start/busy/done FSM interface; reads acc only after done=1.
//
//  Data layout (matches array_controller shift-register addressing):
//    a_flat[r*SIZE + c] = A[r][c]   (row-major)
//    b_flat[k*SIZE + c] = B[k][c]   (row-major, i.e. b_flat[r*SIZE+c]=B[r][c])
//
//  Test 1: A = identity, B = all-2  →  C = all-2
//  Test 2: A[r][c] = r+1, B[r][c] = c+1  →  C[r][c] = SIZE*(r+1)*(c+1)
// ============================================================

#include "verilated.h"
#include "Varray_controller.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

static constexpr int SIZE    = 16;
static constexpr int TIMEOUT = 500;   // max ticks before declaring hang

static Varray_controller* top;
static vluint64_t sim_time = 0;

static void tick() {
    top->clk = 0; top->eval(); sim_time++;
    top->clk = 1; top->eval(); sim_time++;
}

// Load flat matrices and pulse start; wait for done=1; return false on timeout.
static bool run_test(const char*    name,
                     const int8_t   A[SIZE][SIZE],
                     const int8_t   B[SIZE][SIZE],
                     const int32_t  expected[SIZE][SIZE])
{
    printf("\n--- %s ---\n", name);

    // ---- reset -------------------------------------------------------
    top->rst_n = 0; top->start = 0;
    for (int i = 0; i < SIZE * SIZE; i++) { top->a_flat[i] = 0; top->b_flat[i] = 0; }
    tick(); tick();
    top->rst_n = 1;
    tick();   // one idle cycle after reset

    // ---- load matrices -----------------------------------------------
    for (int r = 0; r < SIZE; r++)
        for (int c = 0; c < SIZE; c++) {
            top->a_flat[r * SIZE + c] = (uint8_t)A[r][c];
            top->b_flat[r * SIZE + c] = (uint8_t)B[r][c];
        }

    // ---- pulse start for 1 cycle ------------------------------------
    top->start = 1; tick();
    top->start = 0;

    // ---- wait for done=1 --------------------------------------------
    int elapsed = 0;
    while (!top->done) {
        tick();
        if (++elapsed > TIMEOUT) {
            printf("  TIMEOUT — done never asserted (busy=%d)\n", (int)top->busy);
            return false;
        }
    }
    printf("  done after %d cycles\n", elapsed + 1);

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

int main(int argc, char** argv)
{
    Verilated::commandArgs(argc, argv);
    top = new Varray_controller;

    // ---- Test 1: A = identity, B = all-2  --------------------------
    int8_t  A1[SIZE][SIZE] = {};
    int8_t  B1[SIZE][SIZE];
    int32_t E1[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++) A1[i][i] = 1;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) { B1[i][j] = 2; E1[i][j] = 2; }

    // ---- Test 2: A[r][c] = r+1, B[r][c] = c+1  --------------------
    // C[r][c] = sum_{k} (r+1)*(c+1) = SIZE*(r+1)*(c+1)
    int8_t  A2[SIZE][SIZE];
    int8_t  B2[SIZE][SIZE];
    int32_t E2[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            A2[i][j] = (int8_t)(i + 1);
            B2[i][j] = (int8_t)(j + 1);
            E2[i][j] = SIZE * (i + 1) * (j + 1);
        }

    bool ok = true;
    ok &= run_test("Test1 [ctrl]: I * all-2 => all-2",          A1, B1, E1);
    ok &= run_test("Test2 [ctrl]: A[r][c]=r+1, B[r][c]=c+1",   A2, B2, E2);

    top->final();
    delete top;

    printf("\n%s\n", ok ? "=== ALL TESTS PASSED ===" : "=== SOME TESTS FAILED ===");
    return ok ? 0 : 1;
}
