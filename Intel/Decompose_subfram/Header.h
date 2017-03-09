#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mkl.h>
#include "mkl_service.h"
#include "mkl_dfti.h"
#include <omp.h>
#include <immintrin.h>



/* Define the format to printf MKL_LONG values */
#if !defined(MKL_ILP64)
#define LI "%li"
#else
#define LI "%lli"
#endif

#define NUM_SYM_SUBFRAME 14
#define Num_SC_RB 12
#define DATA_SIZE 14400


MKL_Complex8 ** compose_subframe(MKL_Complex8* data, MKL_Complex8* dmrs_1, MKL_Complex8* dmrs_2, int RB);
void decompose_subframe(MKL_Complex8 ** gride_in, MKL_Complex8 * data, MKL_Complex8* dmrs_1, MKL_Complex8* dmrs_2, int RB);