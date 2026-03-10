/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: minOrMax.c
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:57:03
 */

/* Include Files */
#include "minOrMax.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const float x[375]
 *                float ex[75]
 *                int idx[75]
 * Return Type  : void
 */
void maximum(const float x[375], float ex[75], int idx[75])
{
  int i;
  int j;
  for (j = 0; j < 75; j++) {
    idx[j] = 1;
    ex[j] = x[5 * j];
    for (i = 0; i < 4; i++) {
      float f;
      boolean_T p;
      f = x[(i + 5 * j) + 1];
      if (rtIsNaNF(f)) {
        p = false;
      } else {
        float f1;
        f1 = ex[j];
        if (rtIsNaNF(f1)) {
          p = true;
        } else {
          p = (f1 < f);
        }
      }
      if (p) {
        ex[j] = f;
        idx[j] = i + 2;
      }
    }
  }
}

/*
 * File trailer for minOrMax.c
 *
 * [EOF]
 */
