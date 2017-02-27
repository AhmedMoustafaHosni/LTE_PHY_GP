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
