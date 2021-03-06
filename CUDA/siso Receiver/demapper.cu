/*
% Function:		demapper
% Description:	Maps complex-valued modulation symbols to binary digits using hard decision
% Inputs:		*symbols_R_h:	Real part of the symbols
%				N:				Number of output bits
%				Qm:				Demodulation type (1=bpsk, 2=qpsk, 4=16qam, or 6=64qam)
% Outputs:		*bits_h:		Demodulated bits
By: Ahmad Nour & Mohammed Mostafa
*/

#include "demapper.cuh"

__global__ void Demapper(cufftComplex* symbols_d, Byte *bits_d, int Qm, int numThreads) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Not to run more threads than available data
	if (idx >= numThreads)
		return;

	float symb_real = symbols_d[idx].x;
	float symb_imag = symbols_d[idx].y;

	switch (Qm)
	{
	case 2:					//QPSK
		if(symb_real >= 0)
			bits_d[idx * Qm] = 0;
		else
			bits_d[idx * Qm] = 1;

		if (symb_imag >= 0)
			bits_d[idx * Qm + 1] = 0;
		else
			bits_d[idx * Qm + 1] = 1;
		break;
	case 4:					//QAM16
		if (symb_real < 0)
			bits_d[idx * Qm] = 1;
		else
			bits_d[idx * Qm] = 0;

		if (symb_imag < 0)
			bits_d[idx * Qm + 1] = 1;
		else
			bits_d[idx * Qm + 1] = 0;

		if (fabsf(symb_real) < (2 * rsqrtf(10)))
			bits_d[idx * Qm + 2] = 0;
		else
			bits_d[idx * Qm + 2] = 1;

		if (fabsf(symb_imag) < (2 * rsqrtf(10)))
			bits_d[idx * Qm + 3] = 0;
		else
			bits_d[idx * Qm + 3] = 1;
		break;
	case 6:					//QAM64
		if (symb_real < 0)
			bits_d[idx * Qm] = 1;
		else
			bits_d[idx * Qm] = 0;

		if (symb_imag < 0)
			bits_d[idx * Qm + 1] = 1;
		else
			bits_d[idx * Qm + 1] = 0;

		if (fabsf(symb_real) < (4 * rsqrtf(42)))
			bits_d[idx * Qm + 2] = 0;
		else
			bits_d[idx * Qm + 2] = 1;

		if (fabsf(symb_imag) < (4 * rsqrtf(42)))
			bits_d[idx * Qm + 3] = 0;
		else
			bits_d[idx * Qm + 3] = 1;

		if (fabsf(symb_real) > (2 * rsqrtf(42)) && (fabsf(symb_real) < (6 * rsqrtf(42))))
			bits_d[idx * Qm + 4] = 0;
		else
			bits_d[idx * Qm + 4] = 1;

		if (fabsf(symb_imag) > (2 * rsqrtf(42)) && (fabsf(symb_imag) < (6 * rsqrtf(42))))
			bits_d[idx * Qm + 5] = 0;
		else
			bits_d[idx * Qm + 5] = 1;
		break;
	default:
		break;
	}

}

void demapper(cufftComplex* symbols_d, Byte** bits_d, const int N, int Qm)
{
	//Calc. number of needed threads for calling kernel(s)
	int numThreads = (N / Qm);
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	// block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); // grid size in bloack (min 1)

	//Calling the kernel(s)
	Demapper << < gridDim, blockDim >> > (symbols_d, *bits_d, Qm, numThreads);

}
