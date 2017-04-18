/*
% Function:		decompose_subframe
% Description:	compose the subframe by multiplexing the dmrs signal and data
% Inputs:		*subframe_h			the subframe with data of all ofdm symbols
%				M_pusch_rb			number of resource blocks assigned to the ue
% Outputs:		*complex_data_h:	complex data to be sent in subframe
%				*dmrs_1_h:			demodulation reference signal number 1
%				*dmrs_2_h:			demodulation reference signal number 2
By: Mohammed Mostafa
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



void decompose_subframe(cufftComplex* subframe_d, const int M_pusch_rb, cufftComplex** complex_data_d, cufftComplex** dmrs_1_d, cufftComplex** dmrs_2_d)
{
	int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//compose subframe
	dim3 grid(2, N_symbs_per_subframe, 1);
	dim3 block(M_pusch_sc / 2, 1, 1);
	decompose_subframe << < grid, block >> > (subframe_d, M_pusch_sc, *complex_data_d, *dmrs_1_d, *dmrs_2_d);

}
