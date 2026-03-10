/*
 * Option 1 benchmark — Hand-written C (gate-split weights, combined bias)
 * Uses shared seed-42 test input for cross-option comparison.
 * Build: gcc -O3 -std=c99 -ffast-math -o benchmark_opt1 lstm_model.c main_benchmark.c -lm -I. -I../benchmark
 */
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include "lstm_model.h"
#include "../benchmark/shared_test_input.h"

#define N_WARMUP  50
#define N_RUNS   1000

int main(void)
{
    int32_t predictions[SEQ_LEN];
    int i, t;
    clock_t t0, t1;
    double total_ms, per_run_ms;

    for (i = 0; i < N_WARMUP; i++)
        lstm_seq_to_seq(shared_test_input, SHARED_SEQ_LEN, predictions);

    t0 = clock();
    for (i = 0; i < N_RUNS; i++)
        lstm_seq_to_seq(shared_test_input, SHARED_SEQ_LEN, predictions);
    t1 = clock();

    total_ms  = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;
    per_run_ms = total_ms / N_RUNS;

    printf("Option 1: Hand-written C (gate-split, combined bias)\n");
    printf("  Runs/warmup : %d / %d\n", N_RUNS, N_WARMUP);
    printf("  Total time  : %.4f ms\n", total_ms);
    printf("  Per run     : %.4f ms  (%.2f us)\n", per_run_ms, per_run_ms * 1000.0);
    printf("  Predictions [%d steps]: [", SEQ_LEN);
    for (t = 0; t < SEQ_LEN; t++)
        printf("%d%s", predictions[t], t < SEQ_LEN - 1 ? " " : "]\n");
    return 0;
}
