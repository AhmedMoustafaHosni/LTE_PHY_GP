/*
% Function:		descrambler
% Description:	descramble bits with psuedo random seq.
% Inputs:		bits_h:				Binary bits to descramble
%			c_h:				psuedo random sequence
% Outputs:		*descrambledbits_h:		Descrambled Bits
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

void descrambler(Byte* bits_d, Byte** descrambledbits_d, Byte* c_d, const int N)
{

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = N;
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Calling the kernel(s)
	descrabmler << <gridDim, blockDim >> > (bits_d, *descrambledbits_d, c_d, N);

}
