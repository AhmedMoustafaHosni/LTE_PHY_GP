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
#define     Q_M	      6   // QPSK = 2, 16-QAM = 4 , 64-QAM = 6
#define   H_prime    FRAME_LENGTH/Q_M


// RI bits
#define N_RI_bits  12
#define H_prime_total (H_prime + N_RI_bits)

// Matrix
#define	   C_mux      12		// 12 symbols excluding DMRS symbols
#define	   R_mux	 H_prime_total * Q_M / C_mux
#define	R_prime_mux  R_mux/Q_M
/**********************************************************************************************/

void interleaver(char* bits, char* ri_bits, char* out);