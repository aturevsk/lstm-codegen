/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: lstm_infer_opt4_initialize.c
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:57:03
 */

/* Include Files */
#include "lstm_infer_opt4_initialize.h"
#include "lstm_infer_opt4.h"
#include "lstm_infer_opt4_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void lstm_infer_opt4_initialize(void)
{
  lstm_infer_opt4_emx_init();
  lstm_infer_opt4_init();
  isInitialized_lstm_infer_opt4 = true;
}

/*
 * File trailer for lstm_infer_opt4_initialize.c
 *
 * [EOF]
 */
