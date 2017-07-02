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

void scrambler(Byte* bits_d, Byte** scrambledbits_d, Byte* c_d, const int N)
{

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = N;
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Calling the kernel(s)
	scrabmler << <gridDim, blockDim , 0, stream_default>> > (bits_d, *scrambledbits_d, c_d, N);

}
