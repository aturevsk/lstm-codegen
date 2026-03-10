/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: lstm_infer_opt3.h
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:51:10
 */

#ifndef LSTM_INFER_OPT3_H
#define LSTM_INFER_OPT3_H

/* Include Files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void lstm_infer_opt3(const float X[225], int pred[75]);

void lstm_infer_opt3_emx_init(void);

void lstm_infer_opt3_init(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for lstm_infer_opt3.h
 *
 * [EOF]
 */
