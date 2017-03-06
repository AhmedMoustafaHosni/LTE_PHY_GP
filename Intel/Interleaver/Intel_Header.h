#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <float.h>
#include <mkl.h>
#include "mkl_service.h"
#include "mkl_dfti.h"
#include <omp.h>
#include <immintrin.h>

using namespace std;

/******************************* Interleaver Parameters   *****************************************/
#define FRAME_LENGTH 86400
#define     MOD	      6   // QPSK = 2, 16-QAM = 4 , 64-QAM = 6

// RI bits
#define N_RI_bits  12

//// Matrix
#define	   C_mux      12		// 12 symbols excluding DMRS symbols

/**********************************************************************************************/

void interleaver(char* bits, int bits_length ,char* ri_bits, int ribits_length, char*& out, int& out_length, char mod);