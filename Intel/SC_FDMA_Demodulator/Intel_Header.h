#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mkl.h>
#include "mkl_service.h"
#include "mkl_dfti.h"
#include <omp.h>
#include <immintrin.h>

#define N_sc_rb  12
#define FFT_size  2048
#define prb_offset  0
#define N_cp_L_0  160
#define N_cp_L_else  144
#define N_symbs_per_slot  7


/* Define the format to printf MKL_LONG values */
#if !defined(MKL_ILP64)
#define LI "%li"
#else
#define LI "%lli"
#endif

void SC_FDMA_demod(_MKL_Complex8* pusch_bb, int N_ul_rb, _MKL_Complex8 input_subframe[][1200]);
void SC_FDMA_mod(_MKL_Complex8* pusch_bb, int M_pusch_rb, _MKL_Complex8 input_subframe[][1200]);