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

#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

//Example for timer macros usage:
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n");
//  ...at the very end
//  destroyTimers();

__global__ void descrabmler(Byte *bits_d, Byte *descrambledbits_d, Byte *c_d, int numThreads);
void descrambler(Byte* bits_d, Byte** descrambledbits_d, Byte* c_d, const int N);
