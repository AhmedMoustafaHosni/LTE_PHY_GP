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
#include <cufft.h>

typedef unsigned char Byte;
typedef signed char signedByte;

#define N_sc_rb  12

#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

//Example for timer macros usage:
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n");
//  ...at the very end
//  destroyTimers();

//__global__ void Demapper(cufftComplex *symbols_d, Byte *bits_d, int Qm, int numThreads);
__global__ void Demapper(cufftComplex* symbols_d, Byte *bits_d, int Qm, double coeff, int numThreads);
//void demapper(cufftComplex* symbols_d, Byte** bits_d, const int N, int Qm);
void demapper(cufftComplex* symbols_d, const int M_pusch_rb, Byte** bits_d, const int N, int Qm);

