#pragma once


/*

Function:		 channel_equalization_zf
Description : equalise the channel effecct on the received signal


Inputs :	  modulated_subframe - received subframe without demodulation signal in one vector
			  channel - estimated channel


Outputs :    equalised symbols

// By : Ahmed Moustafa

*/


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


#define N_sc_rb  12
#define N_symbs_per_slot 7
#define N_data_symbs_per_subframe 12					
#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);


void channel_equalization_zf(cufftComplex* subframe_h, const int M_pusch_sc, cufftComplex* channel_h, cufftComplex** equalized_subframe_h);