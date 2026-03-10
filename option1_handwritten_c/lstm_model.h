/**
 * lstm_model.h
 * LSTM seq-to-seq inference header for STM32F746G-Discovery
 *
 * Model: LSTM(input=3, hidden=50, layers=1, batch_first=True) + FC(50->5)
 * Target: ARM Cortex-M7 @ 216 MHz, single-precision FPU (FPv5-D16)
 *
 * Generated for use with weights defined in lstm_weights.h.
 */

#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H

#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Try to include CMSIS-DSP for ARM-optimised math routines.
 * If building on a host machine for testing, this header will not be present;
 * the guard prevents a fatal build error in that case.
 * --------------------------------------------------------------------------- */
#ifdef USE_CMSIS
#include "arm_math.h"
#endif

/* ---------------------------------------------------------------------------
 * Model dimension macros
 * These must match the values exported by generate_weights.py / lstm_weights.h.
 * --------------------------------------------------------------------------- */
#define INPUT_SIZE   3    /**< Number of features per timestep               */
#define HIDDEN_SIZE  50   /**< LSTM hidden (and cell) state dimension         */
#define NUM_CLASSES  5    /**< Output classes from the FC layer               */
#define SEQ_LEN      75   /**< Default sequence length for one inference pass */

/* ---------------------------------------------------------------------------
 * LSTMState
 * Holds the recurrent hidden state h and cell state c between timesteps.
 * Sized for a single-layer LSTM with HIDDEN_SIZE=50.
 * On STM32F746G the 400 bytes sit comfortably in SRAM (320 KB available).
 * --------------------------------------------------------------------------- */
typedef struct {
    float h[HIDDEN_SIZE];  /**< Hidden state vector, h_t                     */
    float c[HIDDEN_SIZE];  /**< Cell  state vector,  c_t                     */
} LSTMState;

/* ---------------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------------- */

/**
 * lstm_init - Zero-initialise an LSTMState (h=0, c=0).
 *
 * Must be called once before the first lstm_step() of each new sequence.
 *
 * @param s  Pointer to the LSTMState to initialise.
 */
void lstm_init(LSTMState *s);

/**
 * lstm_step - Process one input timestep through the LSTM cell.
 *
 * Implements the standard LSTM equations:
 *   i_t = sigmoid(Wi_i * x_t + Wh_i * h_{t-1} + b_i)
 *   f_t = sigmoid(Wi_f * x_t + Wh_f * h_{t-1} + b_f)
 *   g_t = tanh   (Wi_g * x_t + Wh_g * h_{t-1} + b_g)
 *   o_t = sigmoid(Wi_o * x_t + Wh_o * h_{t-1} + b_o)
 *   c_t = f_t * c_{t-1} + i_t * g_t
 *   h_t = o_t * tanh(c_t)
 *
 * Weights are read from ROM (flash) via lstm_weights.h.
 * The function is compiled at -O3 with inline helpers to keep latency low.
 *
 * @param x  Pointer to the current input vector (length INPUT_SIZE=3).
 * @param s  Pointer to the LSTMState (updated in place).
 */
void lstm_step(const float * __restrict__ x, LSTMState * __restrict__ s);

/**
 * lstm_forward - Run lstm_step for every timestep in a sequence.
 *
 * Convenience wrapper that iterates over seq_len timesteps calling
 * lstm_step().  The caller is responsible for calling lstm_init() first
 * if a fresh sequence is intended.
 *
 * @param input    Flat float array, row-major: input[t * INPUT_SIZE + k]
 * @param seq_len  Number of timesteps to process.
 * @param s        Pointer to the LSTMState (updated in place).
 */
void lstm_forward(const float * __restrict__ input,
                  int seq_len,
                  LSTMState * __restrict__ s);

/**
 * fc_forward - Apply the fully-connected classification layer.
 *
 * Computes: logits[j] = dot(fc_W[j], h) + fc_b[j]  for j in [0, NUM_CLASSES).
 * Weights fc_W (NUM_CLASSES x HIDDEN_SIZE) and fc_b (NUM_CLASSES) are stored
 * in flash via lstm_weights.h.
 *
 * @param h       Pointer to the LSTM hidden state vector (length HIDDEN_SIZE).
 * @param logits  Output array of NUM_CLASSES raw scores (caller-allocated).
 */
void fc_forward(const float * __restrict__ h,
                float       * __restrict__ logits);

/**
 * argmax5 - Return the index of the largest element among 5 values.
 *
 * Hand-unrolled for the fixed NUM_CLASSES=5 case; no branch mis-prediction
 * on deeply pipelined cores.
 *
 * @param v  Pointer to an array of exactly 5 floats.
 * @return   Index in [0, 4] of the maximum value.
 */
int32_t argmax5(const float *v);

/**
 * lstm_seq_to_seq - Full seq-to-seq inference: LSTM + FC + argmax per step.
 *
 * This is the main entry point for a complete inference pass.
 *
 * For each timestep t in [0, seq_len):
 *   1. lstm_step(&input[t * INPUT_SIZE], &state)
 *   2. fc_forward(state.h, logits)
 *   3. output[t] = argmax5(logits)
 *
 * @param input    Flat input array, layout: input[t * INPUT_SIZE + k].
 *                 Must contain seq_len * INPUT_SIZE valid floats.
 * @param seq_len  Number of timesteps (typically SEQ_LEN=75).
 * @param output   Caller-allocated array of seq_len int32_t values;
 *                 output[t] receives the predicted class index for timestep t.
 */
void lstm_seq_to_seq(const float   * __restrict__ input,
                     int            seq_len,
                     int32_t       * __restrict__ output);

#endif /* LSTM_MODEL_H */
