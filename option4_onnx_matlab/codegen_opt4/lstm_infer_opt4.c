/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: lstm_infer_opt4.c
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:57:03
 */

/* Include Files */
#include "lstm_infer_opt4.h"
#include "callPredict.h"
#include "lstm_infer_opt4_data.h"
#include "lstm_infer_opt4_emxutil.h"
#include "lstm_infer_opt4_initialize.h"
#include "lstm_infer_opt4_types.h"
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
void lstm_infer_opt4(const float X[225], int pred[75])
{
  float Y_Data[375];
  float ex[75];
  int i;
  if (!isInitialized_lstm_infer_opt4) {
    lstm_infer_opt4_initialize();
  }
  predict(net.InternalState.InternalValue, X, Y_Data);
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
void lstm_infer_opt4_emx_init(void)
{
  c_emxInitStruct_coder_internal_(&net);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void lstm_infer_opt4_init(void)
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
 * File trailer for lstm_infer_opt4.c
 *
 * [EOF]
 */
