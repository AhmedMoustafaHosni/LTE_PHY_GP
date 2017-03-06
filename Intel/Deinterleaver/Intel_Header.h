#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <float.h>
#include <mkl.h>
//#include "mkl_service.h"
//#include "mkl_dfti.h"
//#include <omp.h>
#include <immintrin.h>

using namespace std;

/******************************* Deinterleaver Parameters   *****************************************/
#define FRAME_LENGTH 1728
#define     MOD	      2   // QPSK = 2, 16-QAM = 4 , 64-QAM = 6
/**********************************************************************************************/

/******************************* Interleaver Parameters   *****************************************/
// RI bits
#define N_RI_bits  12

//// Matrix
#define	   C_mux      12		// 12 symbols excluding DMRS symbols

/**********************************************************************************************/

void deinterleaver(char* bits, int bits_length, char mod, char N_ri_bits,  char* out_ri_bits, char* out);
void interleaver(char* bits, int bits_length, char* ri_bits, int ribits_length, char*& out, int& out_length, char mod);