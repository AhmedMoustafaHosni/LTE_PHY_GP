/*
% Function:		decompose_subframe
% Description:	compose the subframe by multiplexing the dmrs signal and data
% Inputs:		*subframe_h			the subframe with data of all ofdm symbols
%				M_pusch_rb			number of resource blocks assigned to the ue
% Outputs:		*complex_data_h:	complex data to be sent in subframe
%				*dmrs_1_h:			demodulation reference signal number 1
%				*dmrs_2_h:			demodulation reference signal number 2
By: Ahmad Nour & Mohammed Mostafa
*/

/*
coeff_multiply kernel just multiples the output symbols by a coeff. The kernel's overhead can be avoided if we
merged it with the mapper kernel
*/

#include "decompose_subframe.cuh"

__global__ void decompose_subframe(cufftComplex* subframe_d, int M_pusch_sc, cufftComplex* complex_data_d, cufftComplex* dmrs_1_d, cufftComplex* dmrs_2_d) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y;

	if (y_idx == 3)
		dmrs_1_d[x_idx] = subframe_d[y_idx*M_pusch_sc + x_idx];
	else if (y_idx == 10)
		dmrs_2_d[x_idx] = subframe_d[y_idx*M_pusch_sc + x_idx];
	else if (y_idx > 10)
		complex_data_d[(y_idx - 2)*M_pusch_sc + x_idx] = subframe_d[y_idx*M_pusch_sc + x_idx];
	else if (y_idx > 3)
		complex_data_d[(y_idx - 1)*M_pusch_sc + x_idx] = subframe_d[y_idx*M_pusch_sc + x_idx];
	else
		complex_data_d[y_idx*M_pusch_sc + x_idx] = subframe_d[y_idx*M_pusch_sc + x_idx];

}



void decompose_subframe(cufftComplex* subframe_h, const int M_pusch_rb, cufftComplex** complex_data_h, cufftComplex** dmrs_1_h, cufftComplex** dmrs_2_h)
{
	int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	//Device data
	cufftComplex* complex_data_d;
	cufftComplex* dmrs_1_d;
	cufftComplex* dmrs_2_d;
	cufftComplex* subframe_d;

	//Host data allocation
	*complex_data_h = (cufftComplex *)malloc(sizeof(cufftComplex)*12*M_pusch_sc);
	*dmrs_1_h = (cufftComplex *)malloc(sizeof(cufftComplex) * M_pusch_sc);
	*dmrs_2_h = (cufftComplex *)malloc(sizeof(cufftComplex) * M_pusch_sc);

	//Device data allocation
	startTimer();
	cudaMalloc((void **)&complex_data_d, sizeof(cufftComplex) * 12 * M_pusch_sc);
	cudaMalloc((void **)&dmrs_1_d, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&dmrs_2_d, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&subframe_d, sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc);
	stopTimer("cudaMalloc Time= %.6f ms\n", elapsed);

	//Copying data to device
	startTimer();
	cudaMemcpy(subframe_d, subframe_h, sizeof(cufftComplex) * N_symbs_per_subframe * M_pusch_sc, cudaMemcpyHostToDevice);
	stopTimer("cudaMemcpy Host->Device Time= %.6f ms\n", elapsed);

	//compose subframe
	dim3 grid(2, N_symbs_per_subframe, 1);
	dim3 block(M_pusch_sc / 2, 1, 1);
	decompose_subframe << < grid, block >> > (subframe_d, M_pusch_sc, complex_data_d, dmrs_1_d, dmrs_2_d);

	//Retrieve data from device
	startTimer();
	cudaMemcpy(*complex_data_h, complex_data_d, sizeof(cufftComplex)*12*M_pusch_sc, cudaMemcpyDeviceToHost);
	cudaMemcpy(*dmrs_1_h, dmrs_1_d, sizeof(cufftComplex)*M_pusch_sc, cudaMemcpyDeviceToHost);
	cudaMemcpy(*dmrs_2_h, dmrs_2_d, sizeof(cufftComplex)*M_pusch_sc, cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);

	// Cleanup
	cudaFree(complex_data_d);
	cudaFree(dmrs_1_d);
	cudaFree(dmrs_2_d);
	cudaFree(subframe_d);

	//Destroy timers
	destroyTimers();
}