#pragma once

extern cudaStream_t stream_default;


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

//__global__ void InitializeLookupTable(double *LookupTable_R_d, double *LookupTable_I_d, int Qm);
//__global__ void Mapper(Byte *bits_d, Byte *bits_each6_d, double *LookupTable_R_d, double *LookupTable_I_d, float *symbols_R_d, float *symbols_I_d, int Qm, int numThreads);
void mapper(Byte* bits_d, const int N, int Qm, const int M_pusch_rb, cufftComplex* symbols_d, Byte *bits_each_Qm_d); // Mohammed
