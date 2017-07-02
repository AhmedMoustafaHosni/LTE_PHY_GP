#pragma once

extern cudaStream_t stream_default;


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

//#define  Transform_Precoder(precoded_data_d,plan_transform_precoder,symbols_d) cufftExecC2C(plan_transform_precoder,symbols_d,*precoded_data_d,CUFFT_FORWARD)

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

//__global__ void toComplex(float* symbols_R_d, float* symbols_I_d, cufftComplex* symbols_d, double coeff, int numThreads);
inline void transform_precoder(cufftComplex** precoded_data_d, cufftHandle plan_transform_precoder, cufftComplex* symbols_d)
{
	cufftSetStream(plan_transform_precoder, stream_default);
	cufftExecC2C(plan_transform_precoder, symbols_d, *precoded_data_d, CUFFT_FORWARD);
}

