/*
% Function:		mapper
% Description:		Maps binary digits to complex-valued modulation symbols
% Inputs:		inputBits:		Binary bits to map
%			Qm:			Modulation type (1=bpsk, 2=qpsk, 4=16qam, or 6=64qam)
% Outputs:		*symbols_R_h:		Real part of the modulation symbols
			*symbols_I_h:		Imag part of the modulation symbols
By: Ahmad Nour & Mohammed Mostafa
*/

#include "mapper.cuh"

__global__ void InitializeLookupTable(double *LookupTable_R_d, double *LookupTable_I_d, int Qm)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	switch (Qm)
	{
	case 2:					//QPSK
		break;
	case 4:					//QAM16
		break;
	case 6:					//QAM64
		double mytable[4] = { (3 * rsqrtf(42)) ,(rsqrtf(42)), (5 * rsqrtf(42)), (7 * rsqrtf(42)) };
		int real_sign, imag_sign;

		if ((idx & 0b100000) == 0)
			real_sign = 1;
		else
			real_sign = -1;

		if ((idx & 0b010000) == 0)
			imag_sign = 1;
		else
			imag_sign = -1;

		LookupTable_R_d[idx] = mytable[((idx & 0b000010) >> 1) + ((idx & 0b001000) >> 2)] * real_sign;
		LookupTable_I_d[idx] = mytable[(idx & 0b000001) + ((idx & 0b000100) >> 1)] * imag_sign;
		break;
	default:
		break;
	}
}

__global__ void Mapper(Byte *bits_d, Byte *bits_each6_d, double *LookupTable_R_d, double *LookupTable_I_d, float *symbols_R_d, float *symbols_I_d, int Qm, int numThreads) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Not to run more threads than available data
	if (idx >= numThreads)
		return;

	switch (Qm)
	{
	case 2:					//QPSK
		break;
	case 4:					//QAM16
		break;
	case 6:					//QAM64
		bits_each6_d[idx] = bits_d[6 * idx] * 32 + bits_d[6 * idx + 1] * 16 + bits_d[6 * idx + 2] * 8 + bits_d[6 * idx + 3] * 4 + bits_d[6 * idx + 4] * 2 + bits_d[6 * idx + 5];

		__syncthreads();

		symbols_R_d[idx] = LookupTable_R_d[bits_each6_d[idx]];
		symbols_I_d[idx] = LookupTable_I_d[bits_each6_d[idx]];
		break;
	default:
		break;
	}
}

void mapper(Byte* bits_h, const int N, int Qm, float** symbols_R_h, float** symbols_I_h)
{
	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	int modOrder = pow(2, Qm);		//Qm = 6 ==> 64QAM ...

	//Device data
	Byte *bits_d, *bits_each6_d;
	float *symbols_R_d, *symbols_I_d;
	double *LookupTable_R_d, *LookupTable_I_d;

	//Host data allocation
	*symbols_R_h = (float *)malloc(sizeof(float)*(N / Qm));
	*symbols_I_h = (float *)malloc(sizeof(float)*(N / Qm));

	//Device data allocation
	startTimer();
	cudaMalloc((void **)&bits_d, sizeof(Byte)*N);
	cudaMalloc((void **)&bits_each6_d, sizeof(Byte)*(N / Qm));
	cudaMalloc((void **)&LookupTable_R_d, sizeof(double)*modOrder);
	cudaMalloc((void **)&LookupTable_I_d, sizeof(double)*modOrder);
	cudaMalloc((void **)&symbols_R_d, sizeof(float)*(N / Qm));
	cudaMalloc((void **)&symbols_I_d, sizeof(float)*(N / Qm));
	stopTimer("cudaMalloc Time= %.6f ms\n", elapsed);

	//Copying data to device
	startTimer();
	cudaMemcpy(bits_d, bits_h, sizeof(Byte)*N, cudaMemcpyHostToDevice);
	stopTimer("cudaMemcpy Host->Device Time= %.6f ms\n", elapsed);

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = (N / Qm);
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Calling the kernel(s)
	startTimer();
	InitializeLookupTable << <1, modOrder >> > (LookupTable_R_d, LookupTable_I_d, Qm);
	Mapper << < gridDim, blockDim >> > (bits_d, bits_each6_d, LookupTable_R_d, LookupTable_I_d, symbols_R_d, symbols_I_d, Qm, numThreads);
	stopTimer("Mapper Time= %.6f ms\n", elapsed);

	//Retrieve data from device
	startTimer();
	cudaMemcpy(*symbols_R_h, symbols_R_d, sizeof(float)*(N / Qm), cudaMemcpyDeviceToHost);
	cudaMemcpy(*symbols_I_h, symbols_I_d, sizeof(float)*(N / Qm), cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);

	//Cleanup
	cudaFree(bits_d);
	cudaFree(bits_each6_d);
	cudaFree(symbols_R_d);
	cudaFree(symbols_I_d);
	cudaFree(LookupTable_R_d);
	cudaFree(LookupTable_I_d);

	//Destroy timers
	destroyTimers();
}
