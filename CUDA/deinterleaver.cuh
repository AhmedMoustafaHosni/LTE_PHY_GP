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

#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

//Example for timer macros usage:
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n");
//  ...at the very end
//  destroyTimers();

__global__ void initializeMatricies(Byte* y_idx_d, Byte* y_mat_d, int N_idx, int N_mat);
__global__ void interleaveRI(Byte* y_idx_d, Byte* y_mat_d, Byte* ri_d, int R_prime_mux, int N_ri_bits);
__global__ void interleaveData(Byte* y_idx_d, Byte* y_mat_d, Byte* input_d, int numThreads, int H_prime_total, int N_ri_bits, int Qm, int N_l);
__global__ void serialOut(Byte* output_d, Byte* y_mat_d, int Nrows, int Qm, int N_l);
void deinterleaver(const Byte* input_h, Byte** ri_h, Byte** output_h, const int N, const int N_ri, const int Qm, const int N_l);

