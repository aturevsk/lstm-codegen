/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: lstm_infer_opt3.c
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:51:10
 */

/* Include Files */
#include "lstm_infer_opt3.h"
#include "callPredict.h"
#include "lstm_infer_opt3_data.h"
#include "lstm_infer_opt3_emxutil.h"
#include "lstm_infer_opt3_initialize.h"
#include "lstm_infer_opt3_types.h"
#include "minOrMax.h"
#include "rt_nonfinite.h"
#include <string.h>

/* Variable Definitions */
static c_coder_internal_ctarget_dlnetw net;

/* Function Definitions */
/*
 * Arguments    : const float X[225]
 *                int pred[75]
 * Return Type  : void
 */
void lstm_infer_opt3(const float X[225], int pred[75])
{
  float Y_Data[375];
  float ex[75];
  int i;
  if (!isInitialized_lstm_infer_opt3) {
    lstm_infer_opt3_initialize();
  }
  /*  lstm_infer_opt3.m  — codegen entry point for Option 3 */
  /*  importNetworkFromPyTorch -> dlnetwork -> MATLAB Coder */
  /*  */
  /*  Input : single [3 x 75]   (Channels x Time, one sample) */
  /*  Output: int32  [1 x 75]   (per-step class index, 0-indexed) */
  /*  dlarray format: CT (Channels x Time), batch-less for codegen */
  predict(net.InternalState.InternalValue, X, Y_Data);
  /*  [5 x 75] float32 logits */
  /*  [5 x 75] float32 */
  /*  Argmax along dim 1 (class axis) -> 1-indexed, convert to 0-indexed */
  maximum(Y_Data, ex, pred);
  for (i = 0; i < 75; i++) {
    int q0;
    q0 = pred[i];
    if (q0 < -2147483647) {
      q0 = MIN_int32_T;
    } else {
      q0--;
    }
    pred[i] = q0;
  }
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void lstm_infer_opt3_emx_init(void)
{
  c_emxInitStruct_coder_internal_(&net);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void lstm_infer_opt3_init(void)
{
  net.InternalState.InternalValue[0].data.size[0] = 50;
  net.InternalState.InternalValue[0].data.size[1] = 1;
  net.InternalState.InternalValue[1].data.size[0] = 50;
  net.InternalState.InternalValue[1].data.size[1] = 1;
  memset(&net.InternalState.InternalValue[0].data.data[0], 0,
         50U * sizeof(float));
  memset(&net.InternalState.InternalValue[1].data.data[0], 0,
         50U * sizeof(float));
}

/*
 * File trailer for lstm_infer_opt3.c
 *
 * [EOF]
 */
