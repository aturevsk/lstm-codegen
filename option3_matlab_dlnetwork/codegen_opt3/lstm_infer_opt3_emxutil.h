/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: lstm_infer_opt3_emxutil.h
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:51:10
 */

#ifndef LSTM_INFER_OPT3_EMXUTIL_H
#define LSTM_INFER_OPT3_EMXUTIL_H

/* Include Files */
#include "lstm_infer_opt3_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void
c_emxInitMatrix_coder_internal_(coder_internal_DataHolder pMatrix[2]);

extern void
c_emxInitStruct_coder_internal_(c_coder_internal_ctarget_dlnetw *pStruct);

extern void
c_emxInitStruct_dltargets_inter(dltargets_internal_NetworkTable *pStruct);

extern void d_emxInitStruct_coder_internal_(coder_internal_DataHolder *pStruct);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for lstm_infer_opt3_emxutil.h
 *
 * [EOF]
 */
