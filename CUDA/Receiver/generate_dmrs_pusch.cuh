#pragma once

#include <cufft.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <string.h>

typedef unsigned char Byte;
typedef signed char signedByte;

#define N_sc_rb  12
#define FFT_size 2048
#define N_cp_L_0 160
#define N_cp_L_else 144
#define N_symbs_per_slot 7
#define N_symbs_per_subframe 14					//2*N_symbs_per_slot
#define modulated_subframe_length 30720			//2*N_cp_L_0 + 12*N_cp_L_else + 14*FFT_size
//#define timerInit(); float elapsed = 0; cudaEvent_t start, stop;
//#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
//#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

#define PI  3.14159265358979323846
#define N_rb_ul_max 110
#define N_ul_symb 7
#define N_sc_rb 12
const int N_1_DMRS[] = { 0,2,3,4,6,8,9,10 };
const int N_2_DMRS_LAMBDA[8][4] = { { 0, 6, 3, 9 },{ 6, 0, 9, 3 },{ 3, 9, 6, 0 },{ 4,10, 7, 1 },{ 2, 8, 5,11 },{ 8, 2,11, 5 },{ 10, 4, 1, 7 },{ 9, 3, 0, 6 } };
const int W_VECTOR[] = {1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1};


//Example for timer macros usage:
//	timerInit();
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n", elapsed);
//  ...at the very end
//  destroyTimers();


//__global__ void generate_reference_signal(cufftComplex* dmrs2_d, int w_vector, int M_sc_rb);
void generate_dmrs_pusch(int N_subfr, int N_id_cell, int delta_ss, bool group_hopping_enabled, bool sequence_hopping_enabled, int cyclic_shift, int cyclic_shift_dci, char* w_config, int N_prbs, int layer, cufftComplex** dmrs1_d, cufftComplex** dmrs2_d, cufftComplex* x_q_d);