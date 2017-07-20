/*
% Function:		transform_precoder
% Description:	perform transform precoding on complex data after mapper
% Inputs:		*symbols_R_h:	Real part of the symbols
% Inputs:		*symbols_I_h:	Imag part of the symbols
%				M_pusch_rb		numer of resource blocks assigned to ue
% Outputs:		*precoded_data	transform precodded data
By: Ahmad Nour & Mohammed Mostafa
*/

/*
toComplex kernel converts from 2-array implementation 2 cufftComplex structure and
multiples symbols by a coeff (1/sqrt(M_pusch_sc)). The kernel's overhead can be avoided if we
merged it with the mapper kernel
*/

#include "transform_precoder.cuh"

__global__ void toComplex(float* symbols_R_d, float* symbols_I_d, cufftComplex* symbols_d, double coeff, int numThreads) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Not to run more threads than available data
	if (idx >= numThreads)
		return;

	symbols_d[idx].x = symbols_R_d[idx] * coeff;
	symbols_d[idx].y = symbols_I_d[idx] * coeff;

}

void transform_precoder(float* symbols_R_d,float* symbols_I_d, const int M_pusch_rb, int signal_size, cufftComplex** precoded_data_d, cufftHandle plan_transform_precoder, cufftComplex* symbols_d)
{
	int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//Calc. number of needed threads for calling kernel(s)
	int numThreads = signal_size;
	int blockDim = (numThreads < 1024) ? numThreads : 1024;	//block size in threads (max 1024 thread)
	int gridDim = numThreads / (blockDim)+(numThreads % blockDim == 0 ? 0 : 1); //grid size in bloack (min 1)

	//Coeff. Multiplication
	toComplex << <gridDim, blockDim >> > (symbols_R_d, symbols_I_d, symbols_d, rsqrtf(M_pusch_sc), numThreads);

	cufftExecC2C(plan_transform_precoder, symbols_d, *precoded_data_d, CUFFT_FORWARD);

}
