#pragma once

extern cudaStream_t stream_default;


#include <helper_math.h>
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <cuda.h>

typedef unsigned char Byte;
typedef signed char signedByte;

#define Nc 1600		// specified by the 36211 v8 standard for random sequence generation   

//#define timerInit(); float elapsed = 0; cudaEvent_t start, stop;
//#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
//#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

//Example for timer macros usage:
//	timerInit();
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n",elapsed);
//  ...at the very end
//  destroyTimers();

//__global__ void scrabmler(Byte *bits_d, Byte *scrambledbits_d, Byte *c_d, int numThreads);
void generate_psuedo_random_seq(Byte** c_h, const int seq_len, const int n_RNTI, const int n_s, const int N_id_cell);