/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: callPredict.h
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:57:03
 */

#ifndef CALLPREDICT_H
#define CALLPREDICT_H

/* Include Files */
#include "lstm_infer_opt4_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void predict(const coder_internal_DataHolder c_dlnet_InternalState_InternalV[2],
             const float inputsT_0_f1[225], float outputs_0_f1[375]);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for callPredict.h
 *
 * [EOF]
 */
