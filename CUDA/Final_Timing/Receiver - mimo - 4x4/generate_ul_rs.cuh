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

#include <string.h>

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
////#define startTimer(); cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
////#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
////#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

#define PI  3.14159265358979323846
#define N_rb_ul_max 110
#define N_sc_rb 12
__device__ const char PHI_1_d[] = { -1, 1, 3,-3, 3, 3, 1, 1, 3, 1,-3, 3,1, 1, 3, 3, 3,-1, 1,-3,-3, 1,-3, 3,1, 1,-3,-3,-3,-1,-3,-3, 1,-3, 1,-1,-1, 1, 1, 1, 1,-1,-3,-3, 1,-3, 3,-1 ,-1, 3, 1,-1, 1,-1,-3,-1, 1,-1, 1, 3 ,1,-3, 3,-1,-1, 1, 1,-1,-1, 3,-3, 1 ,-1, 3,-3,-3,-3, 3, 1,-1, 3, 3,-3, 1 ,-3,-1,-1,-1, 1,-3, 3,-1, 1,-3, 3, 1 ,1,-3, 3, 1,-1,-1,-1, 1, 1, 3,-1, 1 ,1,-3,-1, 3, 3,-1,-3, 1, 1, 1, 1, 1 ,-1, 3,-1, 1, 1,-3,-3,-1,-3,-3, 3,-1 ,3, 1,-1,-1, 3, 3,-3, 1, 3, 1, 3, 3 ,1,-3, 1, 1,-3, 1, 1, 1,-3,-3,-3, 1 ,3, 3,-3, 3,-3, 1, 1, 3,-1,-3, 3, 3 ,-3, 1,-1,-3,-1, 3, 1, 3, 3, 3,-1, 1, 3,-1, 1,-3,-1,-1, 1, 1, 3, 1,-1,-3 ,1, 3, 1,-1, 1, 3, 3, 3,-1,-1, 3,-1 ,-3, 1, 1, 3,-3, 3,-3,-3, 3, 1, 3,-1 ,-3, 3, 1, 1,-3, 1,-3,-3,-1,-1, 1,-3 ,-1, 3, 1, 3, 1,-1,-1, 3,-3,-1,-3,-1 ,-1,-3, 1, 1, 1, 1, 3, 1,-1, 1,-3,-1 ,-1, 3,-1, 1,-3,-3,-3,-3,-3, 1,-1,-3 ,1, 1,-3,-3,-3,-3,-1, 3,-3, 1,-3, 3 ,1, 1,-1,-3,-1,-3, 1,-1, 1, 3,-1, 1 ,1, 1, 3, 1, 3, 3,-1, 1,-1,-3,-3, 1 ,1,-3, 3, 3, 1, 3, 3, 1,-3,-1,-1, 3 ,1, 3,-3,-3, 3,-3, 1,-1,-1, 3,-1,-3 ,-3,-1,-3,-1,-3, 3, 1,-1, 1, 3,-3,-3, -1, 3,-3, 3,-1, 3, 3,-3, 3, 3,-1,-1, 3,-3,-3,-1,-1,-3,-1, 3,-3, 3, 1,-1  };
// phi(n) when 2 resource block assigned for the user
__device__ const char PHI_2_d[] = { -1,3,1,-3,3,-1,1,3,-3,3,1,3,-3,3,1,1,-1,1,3,-3,3,-3,-1,-3,-3,3,-3,-3,-3,1,-3,-3,3,-1,1,1,1,3,1,-1,3,-3,-3,1,3,1,1,-3,3,-1,3,3,1,1,-3,3,3,3,3,1,-1,3,-1,1,1,-1,-3,-1,-1,1,3,3,-1,-3,1,1,3,-3,1,1,-3,-1,-1,1,3,1,3,1,-1,3,1,1,-3,-1,-3,-1,-1,-1,-1,-3,-3,-1,1,1,3,3,-1,3,-1,1,-1,-3,1,-1,-3,-3,1,-3,-1,-1,-3,1,1,3,-1,1,3,1,-3,1,-3,1,1,-1,-1,3,-1,-3,3,-3,-3,-3,1,1,1,1,-1,-1,3,-3,-3,3,-3,1,-1,-1,1,-1,1,1,-1,-3,-1,1,-1,3,-1,-3,-3,3,3,-1,-1,-3,-1,3,1,3,1,3,1,1,-1,3,1,-1,1,3,-3,-1,-1,1,-3,1,3,-3,1,-1,-3,3,-3,3,-1,-1,-1,-1,1,-3,-3,-3,1,-3,-3,-3,1,-3,1,1,-3,3,3,-1,-3,-1,3,-3,3,3,3,-1,1,1,-3,1,-1,1,1,-3,1,1,-1,1,-3,-3,3,-1,3,-1,-1,-3,-3,-3,-1,-3,-3,1,-1,1,3,3,-1,1,-1,3,1,3,3,-3,-3,1,3,1,-1,-3,-3,-3,3,3,-3,3,3,-1,-3,3,-1,1,-3,1,1,3,3,1,1,1,-1,-1,1,-3,3,-1,1,1,-3,3,3,-1,-3,3,-3,-1,-3,-1,3,-1,-1,-1,-1,-3,-1,3,3,1,-1,1,3,3,3,-1,1,1,-3,1,3,-1,-3,3,-3,-3,3,1,3,1,-3,3,1,3,1,1,3,3,-1,-1,-3,1,-3,-1,3,1,1,3,-1,-1,1,-3,1,3,-3,1,-1,-3,-1,3,1,3,1,-1,-3,-3,-1,-1,-3,-3,-3,-1,-1,-3,3,-1,-1,-1,-1,1,1,-3,3,1,3,3,1,-1,1,-3,1,-3,1,1,-3,-1,1,3,-1,3,3,-1,-3,1,-1,-3,3,3,3,-1,1,1,3,-1,-3,-1,3,-1,-1,-1,1,1,1,1,1,-1,3,-1,-3,1,1,3,-3,1,-3,-1,1,1,-3,-3,3,1,1,-3,1,3,3,1,-1,-3,3,-1,3,3,3,-3,1,-1,1,-1,-3,-1,1,3,-1,3,-3,-3,-1,-3,3,-3,-3,-3,-1,-1,-3,-1,-3,3,1,3,-3,-1,3,-1,1,-1,3,-3,1,-1,-3,-3,1,1,-1,1,-1,1,-1,3,1,-3,-1,1,-1,1,-1,-1,3,3,-3,-1,1,-3,-3,-1,-3,3,1,-1,-3,-1,-3,-3,3,-3,3,-3,-1,1,3,1,-3,1,3,3,-1,-3,-1,-1,-1,-1,3,3,3,1,3,3,-3,1,3,-1,3,-1,3,3,-3,3,1,-1,3,3,1,-1,3,3,-1,-3,3,-3,-1,-1,3,-1,3,-1,-1,1,1,1,1,-1,-1,-3,-1,3,1,-1,1,-1,3,-1,3,1,1,-1,-1,-3,1,1,-3,1,3,-3,1,1,-3,-3,-1,-1,-3,-1,1,3,1,1,-3,-1,-1,-3,3,-3,3,1,-3,3,-3,1,-1,1,-3,1,1,1,-1,-3,3,3,1,1,3,-1,-3,-1,-1,-1,3,1,-3,-3,-1,3,-3,-1,-3,-1,-3,-1,-1,-3,-1,-1,1,-3,-1,-1,1,-1,-3,1,1,-3,1,-3,-3,3,1,1,-1,3,-1,-1,1,1,-1,-1,-3,-1,3,-1,3,-1,1,3,1,-1,3,1,3,-3,-3,1,-1,-1,1,3 };
// when more than 3 RB assigned for the user
// adding only the used prime numbers
const int prime_nums[110] = { 11, 23, 31, 47, 59, 71, 83, 89, 107, 113, 131, 139, 151, 167, 179, 191, 199, 211, 227, 239, 251, 263, 271, 283, 293, 311, 317, 331, 347, 359, 367, 383, 389, 401, 419, 431, 443, 449, 467, 479, 491, 503, 509, 523, 523, 547, 563, 571, 587, 599, 607, 619, 631, 647, 659, 661, 683, 691, 701, 719, 727, 743, 751, 761, 773, 787, 797, 811, 827, 839, 839, 863, 863, 887, 887, 911, 919, 929, 947, 953, 971, 983, 991, 997, 1019, 1031, 1039, 1051, 1063, 1069, 1091, 1103, 1109, 1123, 1129, 1151, 1163, 1171, 1187, 1193, 1201, 1223, 1231, 1237, 1259, 1259, 1283, 1291, 1307, 1319 };

//Example for timer macros usage:
//	timerInit();
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n", elapsed);
//  ...at the very end
//  destroyTimers();


//__global__ void calculate_x_q(int q, int N_zc_rs, cufftComplex* x_q_d);
//__global__ void calculate_ref_sig_case1(cufftComplex* x_q_d, int N_zc_rs, float alpha, int M_sc_rs, cufftComplex* ref_signal_d);
//__global__ void calculate_ref_sig_case2(int u, float alpha, int M_sc_rs, char* PHI_1_d, cufftComplex* ref_signal_d);
//__global__ void calculate_ref_sig_case3(int u, float alpha, int M_sc_rs, char* PHI_2_d, cufftComplex* ref_signal_d);

void generate_ul_rs(int N_s, int N_id_cell, char* chan_type, int delta_ss, bool group_hopping_enabled, bool sequence_hopping_enabled, float alpha, int N_prbs, cufftComplex** ref_signal_d, cufftComplex* x_q_d);