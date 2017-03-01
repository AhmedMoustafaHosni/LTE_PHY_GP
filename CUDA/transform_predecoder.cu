/*
% Function:		transform_predecoder
% Description:	perform transform predecoding on complex data after sc-fdma demdulation
% Inputs:		*symbols_h:			complex data output from sc-fdma demdulator
%				M_pusch_rb			numer of resource blocks assigned to ue
% Outputs:		*prdecoded_data	transform predecodded data
By: Ahmad Nour & Mohammed Mostafa
*/

/*
coeff_multiply kernel just multiples the output symbols by a coeff. The kernel's overhead can be avoided if we
merged it with the mapper kernel
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

void transform_predecoder(cufftComplex* symbols_h, const int M_pusch_rb, int signal_size, cufftComplex** precoded_data_h)
{
	int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	//Device data
	cufftComplex* symbols_d;
	cufftComplex* predecoded_data_d;

	//Host data allocation
	*precoded_data_h = (cufftComplex *)malloc(sizeof(cufftComplex)*signal_size);
	
	//Device data allocation
	startTimer();
	cudaMalloc((void **)&symbols_d, sizeof(cufftComplex)*signal_size);
	cudaMalloc((void **)&predecoded_data_d, sizeof(cufftComplex)*signal_size);
	stopTimer("cudaMalloc Time= %.6f ms\n", elapsed);

	//Copying data to device
	startTimer();
	cudaMemcpy(symbols_d, symbols_h, sizeof(cufftComplex)*signal_size, cudaMemcpyHostToDevice);
	stopTimer("cudaMemcpy Host->Device Time= %.6f ms\n", elapsed);

	// CUFFT plan
	int N_SIGS = signal_size / M_pusch_sc;
	int n[1] = { M_pusch_sc };
	cufftHandle plan;

	cufftPlanMany(&plan, 1, n, NULL, 1, M_pusch_sc, NULL, 1, M_pusch_sc, CUFFT_C2C, N_SIGS);
	cufftExecC2C(plan, symbols_d, predecoded_data_d, CUFFT_INVERSE);

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = signal_size;
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Coeff. Multiplication
	coeff_multiply << <gridDim, blockDim >> > (predecoded_data_d, rsqrtf(M_pusch_sc), numThreads);

	//Retrieve data from device
	startTimer();
	cudaMemcpy(*precoded_data_h, predecoded_data_d, sizeof(cufftComplex)*signal_size, cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);

	// Cleanup
	cudaFree(symbols_d);
	cudaFree(predecoded_data_d);

	//Destroy timers
	destroyTimers();
}
