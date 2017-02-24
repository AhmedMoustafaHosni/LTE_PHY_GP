/*
% Function:		descrambler
% Description:	descramble bits with psuedo random seq.
% Inputs:		bits_h:				Binary bits to descramble
%				c_h:				psuedo random sequence
% Outputs:		*descrambledbits_h:	Descrambled Bits
By: Ahmad Nour & Mohammed Mostafa
*/

#include "descrambler.cuh"

__global__ void descrabmler(Byte *bits_d, Byte *descrambledbits_d, Byte *c_d, int numThreads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Not to run more threads than available data
	if (idx >= numThreads)
		return;

	descrambledbits_d[idx] = bits_d[idx] ^ c_d[idx];
}

void descrambler(Byte* bits_h, Byte** descrambledbits_h, const Byte* c_h, const int N)
{
	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	//Device data
	Byte *bits_d;
	Byte *descrambledbits_d;
	Byte *c_d;

	//Host data allocation
	*descrambledbits_h = (Byte *)malloc(sizeof(Byte)*N);

	//Device data allocation
	startTimer();
	cudaMalloc((void **)&bits_d, sizeof(Byte)*N);
	cudaMalloc((void **)&descrambledbits_d, sizeof(Byte)*N);
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
	descrabmler << <gridDim, blockDim >> > (bits_d, descrambledbits_d, c_d, N);
	stopTimer("Scrambler Time= %.6f ms\n", elapsed);

	//Retrieve data from device
	startTimer();
	cudaMemcpy(*descrambledbits_h, descrambledbits_d, sizeof(Byte)*N, cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);

	// Cleanup
	cudaFree(bits_d);
	cudaFree(descrambledbits_d);
	cudaFree(c_d);

	//Destroy timers
	destroyTimers();
}
