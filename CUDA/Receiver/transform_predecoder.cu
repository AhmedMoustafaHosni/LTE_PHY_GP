/*
% Function:		transform_predecoder
% Description:	perform transform predecoding on complex data after sc-fdma demdulation
% Inputs:		*symbols_h:			complex data output from sc-fdma demdulator
%				M_pusch_rb			numer of resource blocks assigned to ue
% Outputs:		*prdecoded_data	transform predecodded data
By: Mohammed Mostafa
*/


#include "transform_predecoder.cuh"

__global__ void coeff_multiply(cufftComplex* symbols_d, double coeff, int numThreads) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Not to run more threads than available data
	if (idx >= numThreads)
		return;

	symbols_d[idx].x *= coeff;
	symbols_d[idx].y *= coeff;
}

void transform_predecoder(cufftComplex* symbols_d, const int M_pusch_rb, int signal_size, cufftComplex** predecoded_data_d, cufftHandle plan_transform_predecoder)
{
	int M_pusch_sc = N_sc_rb * M_pusch_rb;

	// CUFFT plan
	cufftExecC2C(plan_transform_predecoder, symbols_d, *predecoded_data_d, CUFFT_INVERSE);

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = signal_size;
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Coeff. Multiplication
	coeff_multiply << <gridDim, blockDim >> > (*predecoded_data_d, rsqrtf(M_pusch_sc), numThreads);

}
