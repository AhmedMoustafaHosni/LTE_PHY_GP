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
	int real_sign, imag_sign;

	switch (Qm)
	{
	case 2:					//QPSK
		if (idx < 2)
			LookupTable_R_d[idx] = rsqrtf(2);
		else
			LookupTable_R_d[idx] = -1*rsqrtf(2);

		if ( (idx % 2) == 0 )
			LookupTable_I_d[idx] = rsqrtf(2);
		else
			LookupTable_I_d[idx] = -1*rsqrtf(2);
		break;
	case 4:					//QAM16
		double mytable1[2] = { (rsqrtf(10)), (3 * rsqrtf(10)) };

		if ((idx & 0b1000) == 0)
			real_sign = 1;
		else
			real_sign = -1;

		if ((idx & 0b0100) == 0)
			imag_sign = 1;
		else
			imag_sign = -1;

		LookupTable_R_d[idx] = mytable1[((idx & 0b0010) >> 1)] * real_sign;
		LookupTable_I_d[idx] = mytable1[(idx & 0b0001)] * imag_sign;
		break;
	case 6:					//QAM64
		double mytable2[4] = { (3 * rsqrtf(42)) ,(rsqrtf(42)), (5 * rsqrtf(42)), (7 * rsqrtf(42)) };

		if ((idx & 0b100000) == 0)
			real_sign = 1;
		else
			real_sign = -1;

		if ((idx & 0b010000) == 0)
			imag_sign = 1;
		else
			imag_sign = -1;

		LookupTable_R_d[idx] = mytable2[((idx & 0b000010) >> 1) + ((idx & 0b001000) >> 2)] * real_sign;
		LookupTable_I_d[idx] = mytable2[(idx & 0b000001) + ((idx & 0b000100) >> 1)] * imag_sign;
		break;
	default:
		break;
	}
}

__global__ void Mapper(Byte *bits_d, Byte *bits_each_Qm_d, double *LookupTable_R_d, double *LookupTable_I_d, float *symbols_R_d, float *symbols_I_d, int Qm, int numThreads) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Not to run more threads than available data
	if (idx >= numThreads)
		return;

	if (Qm == 2)			//QPSK
		bits_each_Qm_d[idx] = bits_d[Qm * idx] * 2 + bits_d[Qm * idx + 1];
	if (Qm == 4)			//QAM16
		bits_each_Qm_d[idx] = bits_d[Qm * idx] * 8 + bits_d[Qm * idx + 1] * 4 + bits_d[Qm * idx + 2] * 2 + bits_d[Qm * idx + 3];
	if (Qm == 6)			//QAM64
		bits_each_Qm_d[idx] = bits_d[Qm * idx] * 32 + bits_d[Qm * idx + 1] * 16 + bits_d[Qm * idx + 2] * 8 + bits_d[Qm * idx + 3] * 4 + bits_d[Qm * idx + 4] * 2 + bits_d[Qm * idx + 5];

	__syncthreads();

	symbols_R_d[idx] = LookupTable_R_d[bits_each_Qm_d[idx]];
	symbols_I_d[idx] = LookupTable_I_d[bits_each_Qm_d[idx]];
}

void mapper(Byte* bits_h, const int N, int Qm, float** symbols_R_h, float** symbols_I_h)
{
	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	int modOrder = pow(2, Qm);		//Qm = 6 ==> 64QAM ...

									//Device data
	Byte *bits_d, *bits_each_Qm_d;
	float *symbols_R_d, *symbols_I_d;
	double *LookupTable_R_d, *LookupTable_I_d;

	//Host data allocation
	*symbols_R_h = (float *)malloc(sizeof(float)*(N / Qm));
	*symbols_I_h = (float *)malloc(sizeof(float)*(N / Qm));

	//Device data allocation
	startTimer();
	cudaMalloc((void **)&bits_d, sizeof(Byte)*N);
	cudaMalloc((void **)&bits_each_Qm_d, sizeof(Byte)*(N / Qm));
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
	Mapper << < gridDim, blockDim >> > (bits_d, bits_each_Qm_d, LookupTable_R_d, LookupTable_I_d, symbols_R_d, symbols_I_d, Qm, numThreads);
	stopTimer("Mapper Time= %.6f ms\n", elapsed);

	//Retrieve data from device
	startTimer();
	cudaMemcpy(*symbols_R_h, symbols_R_d, sizeof(float)*(N / Qm), cudaMemcpyDeviceToHost);
	cudaMemcpy(*symbols_I_h, symbols_I_d, sizeof(float)*(N / Qm), cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);

	//Cleanup
	cudaFree(bits_d);
	cudaFree(bits_each_Qm_d);
	cudaFree(symbols_R_d);
	cudaFree(symbols_I_d);
	cudaFree(LookupTable_R_d);
	cudaFree(LookupTable_I_d);

	//Destroy timers
	destroyTimers();
}
