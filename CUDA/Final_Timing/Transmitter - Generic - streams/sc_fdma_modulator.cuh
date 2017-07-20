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

//Example for timer macros usage:
//	timerInit();
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n", elapsed);
//  ...at the very end
//  destroyTimers();

//__global__ void reshape_ifft_vec(cufftComplex* subframe_d, cufftComplex* ifft_vec_d, int M_pusch_sc);
//__global__ void add_cyclic_prefix(cufftComplex* ifft_vec_d, cufftComplex* pusch_bb_d, int M_pusch_sc);
void sc_fdma_modulator(cufftComplex* subframe_d, const int M_pusch_rb, cufftComplex** pusch_bb_d, cufftHandle plan_sc_fdma, cufftComplex* ifft_vec_d, cudaStream_t stream, int i);