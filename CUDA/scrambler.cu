/*
% Function:		scrambler
% Description:	scramble bits with psuedo random seq.
% Inputs:		bits_h:				Binary bits to scramble
%				c_h:					psuedo random sequence
% Outputs:		*scrambledbits_h:	Scrambled Bits
By: Ahmad Nour & Mohammed Mostafa
*/

#include "scrambler.cuh"

__global__ void scrabmler(Byte *bits_d, Byte *scrambledbits_d, Byte *c_d, int numThreads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Not to run more threads than available data
	if (idx >= numThreads)
		return;

	scrambledbits_d[idx] = bits_d[idx] ^ c_d[idx];
}

void scrambler(Byte* bits_h, Byte** scrambledbits_h, const Byte* c_h, const int N)
{
	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	//Device data
	Byte *bits_d;
	Byte *scrambledbits_d;
	Byte *c_d;

	//Host data allocation
	*scrambledbits_h = (Byte *)malloc(sizeof(Byte)*N);

	//Device data allocation
	startTimer();
	cudaMalloc((void **)&bits_d, sizeof(Byte)*N);
	cudaMalloc((void **)&scrambledbits_d, sizeof(Byte)*N);
	cudaMalloc((void **)&c_d, sizeof(Byte)*N);
	stopTimer("cudaMalloc Time= %.6f ms\n", elapsed);

	//Copying data to device
	startTimer();
	cudaMemcpy(bits_d, bits_h, sizeof(Byte)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c_h, sizeof(Byte)*N, cudaMemcpyHostToDevice);
	stopTimer("cudaMemcpy Host->Device Time= %.6f ms\n", elapsed);

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = N;
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Calling the kernel(s)
	startTimer();
	scrabmler << <gridDim, blockDim >> > (bits_d, scrambledbits_d, c_d, N);
	stopTimer("Scrambler Time= %.6f ms\n", elapsed);

	//Retrieve data from device
	startTimer();
	cudaMemcpy(*scrambledbits_h, scrambledbits_d, sizeof(Byte)*N, cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);

	// Cleanup
	cudaFree(bits_d);
	cudaFree(scrambledbits_d);
	cudaFree(c_d);

	//Destroy timers
	destroyTimers();
}
