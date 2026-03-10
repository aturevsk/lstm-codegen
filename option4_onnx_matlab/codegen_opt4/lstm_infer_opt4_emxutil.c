/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: lstm_infer_opt4_emxutil.c
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:57:03
 */

/* Include Files */
#include "lstm_infer_opt4_emxutil.h"
#include "lstm_infer_opt4_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : coder_internal_DataHolder pMatrix[2]
 * Return Type  : void
 */
void c_emxInitMatrix_coder_internal_(coder_internal_DataHolder pMatrix[2])
{
  int i;
  for (i = 0; i < 2; i++) {
    d_emxInitStruct_coder_internal_(&pMatrix[i]);
  }
}

/*
 * Arguments    : c_coder_internal_ctarget_dlnetw *pStruct
 * Return Type  : void
 */
void c_emxInitStruct_coder_internal_(c_coder_internal_ctarget_dlnetw *pStruct)
{
  c_emxInitStruct_dltargets_inter(&pStruct->InternalState);
}

/*
 * Arguments    : dltargets_internal_NetworkTable *pStruct
 * Return Type  : void
 */
void c_emxInitStruct_dltargets_inter(dltargets_internal_NetworkTable *pStruct)
{
  c_emxInitMatrix_coder_internal_(pStruct->InternalValue);
}

/*
 * Arguments    : coder_internal_DataHolder *pStruct
 * Return Type  : void
 */
void d_emxInitStruct_coder_internal_(coder_internal_DataHolder *pStruct)
{
  pStruct->data.size[0] = 0;
  pStruct->data.size[1] = 0;
}

/*
 * File trailer for lstm_infer_opt4_emxutil.c
 *
 * [EOF]
 */
