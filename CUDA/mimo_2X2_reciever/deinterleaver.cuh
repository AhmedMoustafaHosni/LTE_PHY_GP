#pragma once

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

#define N_pusch_symbs 12;

//#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
//#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

//Example for timer macros usage:
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n");
//  ...at the very end
//  destroyTimers();


//void deinterleaver(Byte* input_d, Byte** ri_d, Byte** output_d, const int N, const int N_ri, const int Qm, const int N_l, Byte* y_idx_d, Byte* y_mat_d);
void deinterleaver(Byte* input_d, Byte* input_d2, Byte** ri_d, Byte** output_d, const int N, const int N_ri, const int Qm, const int N_l, Byte* y_idx_d, Byte* y_mat_d);
__global__ void deinterleaveData(Byte* y_idx_d, Byte* y_mat_d, Byte* output_d, int numThreads, int H_prime_total, int N_ri_bits, int Qm, int N_l);
