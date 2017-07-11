/*

Function:		 channel_equalization_zf
Description : equalise the channel effecct on the received signal


Inputs :	  modulated_subframe - received subframe without demodulation signal in one vector
			  channel - estimated channel


Outputs :    equalised symbols

// By : Ahmed Moustafa

*/

#include "channel_equalization_zf.cuh"

__global__ void channel_equalization_zf_l(cufftComplex* subframe_d, cufftComplex*  channel_d, cufftComplex*  equalized_subframe_d, const int M_pusch_sc)
{
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y;


	equalized_subframe_d[y_idx*M_pusch_sc + x_idx] = cuCdivf(subframe_d[y_idx*M_pusch_sc + x_idx], channel_d[x_idx]);
}



//void channel_equalization_zf(cufftComplex* subframe_h, const int M_pusch_sc, cufftComplex* channel_h, cufftComplex** equalized_subframe_h)
void channel_equalization_zf(cufftComplex* subframe_d, const int M_pusch_sc, cufftComplex* channel_d, cufftComplex** equalized_subframe_d)
{
	
	//Calling the kernel(s)

	
	dim3 grid(2, N_data_symbs_per_subframe, 1);
	dim3 block(M_pusch_sc / 2, 1, 1);
	channel_equalization_zf_l << < grid, block >> > (subframe_d, channel_d, *equalized_subframe_d, M_pusch_sc);
	

	// Cleanup
	//cudaFree(subframe_d);
	//cudaFree(channel_d);
	//cudaFree(equalized_subframe_d);


}
