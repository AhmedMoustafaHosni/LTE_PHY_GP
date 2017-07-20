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
#include <cublas_v2.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))



typedef unsigned char Byte;
typedef signed char signedByte;

#define N_sc_rb  12
#define N_symbs_per_slot 7
#define N_data_symbs_per_subframe 12					
//#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
//#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);


//Example for timer macros usage:
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n");
//  ...at the very end
//  destroyTimers();


//void channe_estimation(cufftComplex* symb_0_h, cufftComplex* symb_1_h, cufftComplex* dmrs_0_h, cufftComplex* dmrs_1_h, const int M_pusch_sc, cufftComplex** channel_h);
void channe_estimation(cufftComplex* symb_0_d, cufftComplex* symb_1_d, cufftComplex* dmrs_0_d, cufftComplex* dmrs_1_d, const int M_pusch_sc, cufftComplex** channel_d);

__global__ void divide_pilots(cufftComplex* symb_0_d, cufftComplex* symb_1_d, cufftComplex* dmrs_0_d, cufftComplex* dmrs_1_d, cufftComplex* filtered_pilots1_d, cufftComplex* filtered_pilots2_d, const int M_pusch_sc);
