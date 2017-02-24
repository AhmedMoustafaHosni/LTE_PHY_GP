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

__global__ void Demapper(float *symbols_R_d, float *symbols_I_d, Byte *bits_d, int Qm, int numThreads) {

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
		float symb_real = symbols_R_d[idx];
		float symb_imag = symbols_I_d[idx];

		if (symb_real < 0)
			bits_d[idx * 6] = 1;
		else
			bits_d[idx * 6] = 0;

		if (symb_imag < 0)
			bits_d[idx * 6 + 1] = 1;
		else
			bits_d[idx * 6 + 1] = 0;

		if (fabsf(symb_real) < (4 * rsqrtf(42)))
			bits_d[idx * 6 + 2] = 0;
		else
			bits_d[idx * 6 + 2] = 1;

		if (fabsf(symb_imag) < (4 * rsqrtf(42)))
			bits_d[idx * 6 + 3] = 0;
		else
			bits_d[idx * 6 + 3] = 1;

		if (fabsf(symb_real) > (2 * rsqrtf(42)) && (fabsf(symb_real) < (6 * rsqrtf(42))))
			bits_d[idx * 6 + 4] = 0;
		else
			bits_d[idx * 6 + 4] = 1;

		if (fabsf(symb_imag) > (2 * rsqrtf(42)) && (fabsf(symb_imag) < (6 * rsqrtf(42))))
			bits_d[idx * 6 + 5] = 0;
		else
			bits_d[idx * 6 + 5] = 1;
		break;
	default:
		break;
	}

}

void demapper(float* symbols_R_h, float* symbols_I_h, Byte** bits_h, const int N, int Qm)
{
	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	//Device data
	Byte *bits_d;
	float *symbols_R_d, *symbols_I_d;

	//Host data allocation
	*bits_h = (Byte *)malloc(sizeof(Byte)*N);

	//Device data allocation
	startTimer();
	cudaMalloc((void **)&symbols_R_d, sizeof(float)*(N / Qm));
	cudaMalloc((void **)&symbols_I_d, sizeof(float)*(N / Qm));
	cudaMalloc((void **)&bits_d, sizeof(Byte)*N);
	stopTimer("cudaMalloc Time= %.6f ms\n", elapsed);

	//Copying data to device
	startTimer();
	cudaMemcpy(symbols_R_d, symbols_R_h, sizeof(float)*(N / Qm), cudaMemcpyHostToDevice);
	cudaMemcpy(symbols_I_d, symbols_I_h, sizeof(float)*(N / Qm), cudaMemcpyHostToDevice);
	stopTimer("cudaMemcpy Host->Device Time= %.6f ms\n", elapsed);

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = (N / Qm);
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	// block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); // grid size in bloack (min 1)

	//Calling the kernel(s)
	startTimer();
	Demapper << < gridDim, blockDim >> > (symbols_R_d, symbols_I_d, bits_d, Qm, numThreads);
	stopTimer("Demapper Time= %.6f ms\n", elapsed);

	//Retrieve data from device
	startTimer();
	cudaMemcpy(*bits_h, bits_d, sizeof(Byte)*N, cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);

	// Cleanup
	cudaFree(bits_d);
	cudaFree(symbols_R_d);
	cudaFree(symbols_I_d);

	//Destroy timers
	destroyTimers();
}
