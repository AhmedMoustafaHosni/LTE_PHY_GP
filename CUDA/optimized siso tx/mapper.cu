/*
% Function:			mapper
% Description:		Maps binary digits to complex-valued modulation symbols
% Inputs:			inputBits:			Binary bits to map
%					Qm:					Modulation type (1=bpsk, 2=qpsk, 4=16qam, or 6=64qam)
% Outputs:			*symbols_R_h:		Real part of the modulation symbols
					*symbols_I_h:		Imag part of the modulation symbols
By: Ahmad Nour & Mohammed Mostafa
Modified by Mohammed Osama
*/

#include "mapper.cuh"

__device__ const float LookupTable_QPSK_R_d[] = { 0.707106781186548,0.707106781186548,-0.707106781186548,-0.707106781186548 };
__device__ const float LookupTable_QPSK_I_d[] = { 0.707106781186548,-0.707106781186548,0.707106781186548,-0.707106781186548 };
__device__ const float LookupTable_16QAM_R_d[] = { 0.316227766016838,0.316227766016838,0.948683298050514,0.948683298050514,0.316227766016838,0.316227766016838,0.948683298050514,0.948683298050514,-0.316227766016838,-0.316227766016838,-0.948683298050514,-0.948683298050514,-0.316227766016838,-0.316227766016838,-0.948683298050514,-0.948683298050514 };
__device__ const float LookupTable_16QAM_I_d[] = { 0.316227766016838,0.948683298050514,0.316227766016838,0.948683298050514,-0.316227766016838,-0.948683298050514,-0.316227766016838,-0.948683298050514,0.316227766016838,0.948683298050514,0.316227766016838,0.948683298050514,-0.316227766016838,-0.948683298050514,-0.316227766016838,-0.948683298050514 };
__device__ const float LookupTable_64QAM_R_d[] = { 0.462910049886276,0.462910049886276,0.154303349962092,0.154303349962092,0.462910049886276,0.462910049886276,0.154303349962092,0.154303349962092,0.771516749810460,0.771516749810460,1.08012344973464,1.08012344973464,0.771516749810460,0.771516749810460,1.08012344973464,1.08012344973464,0.462910049886276,0.462910049886276,0.154303349962092,0.154303349962092,0.462910049886276,0.462910049886276,0.154303349962092,0.154303349962092,0.771516749810460,0.771516749810460,1.08012344973464,1.08012344973464,0.771516749810460,0.771516749810460,1.08012344973464,1.08012344973464,-0.462910049886276,-0.462910049886276,-0.154303349962092,-0.154303349962092,-0.462910049886276,-0.462910049886276,-0.154303349962092,-0.154303349962092,-0.771516749810460,-0.771516749810460,-1.08012344973464,-1.08012344973464,-0.771516749810460,-0.771516749810460,-1.08012344973464,-1.08012344973464,-0.462910049886276,-0.462910049886276,-0.154303349962092,-0.154303349962092,-0.462910049886276,-0.462910049886276,-0.154303349962092,-0.154303349962092,-0.771516749810460,-0.771516749810460,-1.08012344973464,-1.08012344973464,-0.771516749810460,-0.771516749810460,-1.08012344973464,-1.08012344973464 };
__device__ const float LookupTable_64QAM_I_d[] = { 0.462910049886276,0.154303349962092,0.462910049886276,0.154303349962092,0.771516749810460,1.08012344973464,0.771516749810460,1.08012344973464,0.462910049886276,0.154303349962092,0.462910049886276,0.154303349962092,0.771516749810460,1.08012344973464,0.771516749810460,1.08012344973464,-0.462910049886276,-0.154303349962092,-0.462910049886276,-0.154303349962092,-0.771516749810460,-1.08012344973464,-0.771516749810460,-1.08012344973464,-0.462910049886276,-0.154303349962092,-0.462910049886276,-0.154303349962092,-0.771516749810460,-1.08012344973464,-0.771516749810460,-1.08012344973464,0.462910049886276,0.154303349962092,0.462910049886276,0.154303349962092,0.771516749810460,1.08012344973464,0.771516749810460,1.08012344973464,0.462910049886276,0.154303349962092,0.462910049886276,0.154303349962092,0.771516749810460,1.08012344973464,0.771516749810460,1.08012344973464,-0.462910049886276,-0.154303349962092,-0.462910049886276,-0.154303349962092,-0.771516749810460,-1.08012344973464,-0.771516749810460,-1.08012344973464,-0.462910049886276,-0.154303349962092,-0.462910049886276,-0.154303349962092,-0.771516749810460,-1.08012344973464,-0.771516749810460,-1.08012344973464 };

//__global__ void Mapper(Byte *bits_d, Byte *bits_each_Qm_d, float *symbols_R_d, float *symbols_I_d, int Qm, int numThreads) {

__global__ void Mapper(Byte *bits_d, Byte *bits_each_Qm_d, cufftComplex* symbols_d, double coeff, int Qm, int numThreads) { //Mohammed

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Not to run more threads than available data
	if (idx >= numThreads)
		return;
	int index = Qm * idx;  //index of required bits 
	switch (Qm)
	{
	case 2:					//QPSK
		bits_each_Qm_d[idx] = bits_d[index] * 2 + bits_d[index + 1];
		__syncthreads();
		//symbols_R_d[idx] = LookupTable_QPSK_R_d[bits_each_Qm_d[idx]];
		//symbols_I_d[idx] = LookupTable_QPSK_I_d[bits_each_Qm_d[idx]];
		symbols_d[idx].x = LookupTable_QPSK_R_d[bits_each_Qm_d[idx]] * coeff;
		symbols_d[idx].y = LookupTable_QPSK_I_d[bits_each_Qm_d[idx]] * coeff;
		break;
	
	case 4:					//QAM16
		bits_each_Qm_d[idx] = bits_d[index] * 8 + bits_d[index + 1] * 4 + bits_d[index + 2] * 2 + bits_d[index + 3];
		__syncthreads();
		//symbols_R_d[idx] = LookupTable_16QAM_R_d[bits_each_Qm_d[idx]];
		//symbols_I_d[idx] = LookupTable_16QAM_I_d[bits_each_Qm_d[idx]];
		symbols_d[idx].x = LookupTable_16QAM_R_d[bits_each_Qm_d[idx]] * coeff;
		symbols_d[idx].y = LookupTable_16QAM_I_d[bits_each_Qm_d[idx]] * coeff; 
		break;

	case 6:					//QAM64
		bits_each_Qm_d[idx] = bits_d[index] * 32 + bits_d[index + 1] * 16 + bits_d[index + 2] * 8 + bits_d[index + 3] * 4 + bits_d[index + 4] * 2 + bits_d[index + 5];
		__syncthreads();
		//symbols_R_d[idx] = LookupTable_64QAM_R_d[bits_each_Qm_d[idx]];
		//symbols_I_d[idx] = LookupTable_64QAM_I_d[bits_each_Qm_d[idx]];
		symbols_d[idx].x = LookupTable_64QAM_R_d[bits_each_Qm_d[idx]] * coeff;
		symbols_d[idx].y = LookupTable_64QAM_I_d[bits_each_Qm_d[idx]] * coeff; 

		break;
	default:
		break;
	}

}

//void mapper(Byte* bits_d, const int N, int Qm, float** symbols_R_d, float** symbols_I_d, Byte *bits_each_Qm_d)
void mapper(Byte* bits_d, const int N, int Qm, const int M_pusch_rb, cufftComplex* symbols_d, Byte *bits_each_Qm_d) // Mohammed
{
	int modOrder = pow(2, Qm);		//Qm = 6 ==> 64QAM ...
	int M_pusch_sc = N_sc_rb * M_pusch_rb;
	//Calc. number of needed threads for calling kernel(s)
	int numThreads = (N / Qm);
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Calling the kernel(s)
	//Mapper << < gridDim, blockDim >> > (bits_d, bits_each_Qm_d, *symbols_R_d, *symbols_I_d, Qm, numThreads);
		Mapper << < gridDim, blockDim, 0, stream_default>> > (bits_d, bits_each_Qm_d, symbols_d, rsqrtf(M_pusch_sc), Qm, numThreads); //Mohammed
}
