#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <float.h>
#include <mkl.h>
#include "mkl_service.h"
#include "mkl_dfti.h"
#include <immintrin.h>

using namespace std;

/******************************* Input bits Parameters   *****************************************/
#define RESOURCE_BLOCKS 100
#define     MOD	         6              // QPSK = 2, 16-QAM = 4 , 64-QAM = 6


#define M_PUSCH_SC  RESOURCE_BLOCKS*12    // 100 RBs * 12 Subcarriers = 1200 subcarriers 
#define DATA_SIZE   M_PUSCH_SC*12         // 1200 * 12 = 14400 symbols in the grid
#define FRAME_LENGTH  DATA_SIZE*MOD       // 14400*6 = 86400 bits 
//#define FRAME_LENGTH    86400

/**********************************************************************************************/


/******************************* Interleaver Parameters   *****************************************/
// RI bits
#define N_RI_bits  12		// Control bits is a multiple of 12

// Matrix
#define	   C_mux   12		// 12 symbols excluding DMRS symbols
#define INTRLV  FRAME_LENGTH-N_RI_bits*MOD
/**********************************************************************************************/

/******************************* Scrambler Parameters   *****************************************/

#define DataTypeLength 8 // We can divide int to 16 or 8 elements in 256b vector
#define LEN 8            // for accesing the correct elements in the for loop

/**********************************************************************************************/

/******************************* Mapper Parameters   *****************************************/

#define sqrt2 1.41421356237   
#define sqrt2_rec 0.70710678118  // 1/sqrt(2)
//#define NORM_64_6 0.9258200998   // 6/sqrt(42)
#define NORM_64_4 0.6172133998   // 4/sqrt(42)
#define NORM_64_3 0.4629100499   // 3/sqrt(42)
#define NORM_64_2 0.3086066999   // 2/sqrt(42)
//#define NORM_64_1 0.15430335     // 1/sqrt(42)

/**********************************************************************************************/


/******************************* pseudo-random Parameters   *****************************************/

#define Nc  1600        // specified by the 36211 v8 standard for random sequence generation
#define n_RNTI  10      // radio network temporary identifier given to the UE by enodeB(assume 10)

/**********************************************************************************************/

/******************************* DMRS Generation Parameters   *****************************************/

#define N_SC_RB 12            // number of subcarriers per resource block
#define M_PI_6  0.5235987756   // pi/6
#define M_PI    3.14159265358979323846   // pi
#define M_PI_4  0.785398163397448309616  // pi/4
#define N_ZC_RS_MAX 1193				 // largest possible zadoff-chu sequence length for 1200 subcarriers
#define M_RS_SC_MAX 1200

/**********************************************************************************************/


/******************************* Compose subframe Parameters   *****************************************/

#define NUM_SYM_SUBFRAME 14
#define Num_SC_RB 12

/**********************************************************************************************/

/******************************* SCFDMA Modulator Parameters   *****************************************/

#define N_sc_rb  12
#define FFT_size  2048
#define prb_offset  0
#define N_cp_L_0  160
#define N_cp_L_else  144
#define N_symbs_per_slot  7

/**********************************************************************************************/

void interleaver(float* bits, int bits_length, float* ri_bits, int ribits_length, float*& out, int& out_length, char mod);

void mapper(float* bits, int bits_length, MKL_Complex8* symbols, char mod);

float* pseudo_random_sequence_gen(int, int);

void generate_dmrs(unsigned char N_suframe, unsigned int cell_id, unsigned char delta_ss, unsigned char cyclic_shift, unsigned char cyclic_shift_dci, unsigned char N_RB, MKL_Complex8*& dmrs_1, MKL_Complex8*& dmrs_2);

MKL_LONG Transform_precoder(MKL_Complex8*, int);

MKL_Complex8** compose_subframe(MKL_Complex8* data, MKL_Complex8* dmrs_1, MKL_Complex8* dmrs_2, int RB);

void SC_FDMA_mod(_MKL_Complex8* pusch_bb, int M_pusch_rb, _MKL_Complex8** input_subframe);