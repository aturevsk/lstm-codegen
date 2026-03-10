/*
 * Prerelease License - for engineering feedback and testing purposes
 * only. Not for sale.
 * File: lstm_infer_opt4_types.h
 *
 * MATLAB Coder version            : 26.1
 * C/C++ source code generated on  : 09-Mar-2026 21:57:03
 */

#ifndef LSTM_INFER_OPT4_TYPES_H
#define LSTM_INFER_OPT4_TYPES_H

/* Include Files */
#include "rtwtypes.h"

/* Type Definitions */
#ifndef struct_emxArray_real32_T_50x1
#define struct_emxArray_real32_T_50x1
struct emxArray_real32_T_50x1 {
  float data[50];
  int size[2];
};
#endif /* struct_emxArray_real32_T_50x1 */
#ifndef typedef_emxArray_real32_T_50x1
#define typedef_emxArray_real32_T_50x1
typedef struct emxArray_real32_T_50x1 emxArray_real32_T_50x1;
#endif /* typedef_emxArray_real32_T_50x1 */

#ifndef c_typedef_coder_internal_DataHo
#define c_typedef_coder_internal_DataHo
typedef struct {
  emxArray_real32_T_50x1 data;
} coder_internal_DataHolder;
#endif /* c_typedef_coder_internal_DataHo */

#ifndef c_typedef_dltargets_internal_Ne
#define c_typedef_dltargets_internal_Ne
typedef struct {
  coder_internal_DataHolder InternalValue[2];
} dltargets_internal_NetworkTable;
#endif /* c_typedef_dltargets_internal_Ne */

#ifndef c_typedef_c_coder_internal_ctar
#define c_typedef_c_coder_internal_ctar
typedef struct {
  dltargets_internal_NetworkTable InternalState;
} c_coder_internal_ctarget_dlnetw;
#endif /* c_typedef_c_coder_internal_ctar */

#endif
/*
 * File trailer for lstm_infer_opt4_types.h
 *
 * [EOF]
 */
