#pragma once

#include "input.cuh"
#include "generate_psuedo_random_seq.cuh"
#include "interleaver.cuh"
#include "scrambler.cuh"
#include "mapper.cuh"
#include "transform_precoder.cuh"
#include "generate_dmrs_pusch.cuh"
#include "generate_ul_rs.cuh"
#include "compose_subframe.cuh"
#include "sc_fdma_modulator.cuh"

cudaStream_t stream_default;



#define timerInit(); float elapsed = 0; cudaEvent_t start, stop;
#define startTimer(); elapsed = 0; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define stopTimer(msg, var); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf(msg,var);
#define destroyTimers(); 	cudaEventDestroy(start); cudaEventDestroy(stop);

//Example for timer macros usage:
//	timerInit();
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n", elapsed);
//  ...at the very end
//  destroyTimers();