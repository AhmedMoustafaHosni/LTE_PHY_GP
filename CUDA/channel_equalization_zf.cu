/*

Function:		 channel_equalization_zf
Description : equalise the channel effecct on the received signal


Inputs :	  modulated_subframe - received subframe without demodulation signal in one vector
			  channel - estimated channel


Outputs :    equalised symbols

// By : Ahmed Moustafa

*/

#include "channel_equalization_zf.cuh"

__global__ void channel_equalization_zf(cufftComplex* subframe_d, cufftComplex*  channel_d, cufftComplex*  equalized_subframe_d, const int M_pusch_sc)
{
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y;

	equalized_subframe_d[y_idx*M_pusch_sc + x_idx] = cuCdivf(subframe_d[y_idx*M_pusch_sc + x_idx], channel_d[x_idx]);
}



void channel_equalization_zf(cufftComplex* subframe_h, const int M_pusch_sc, cufftComplex* channel_h, cufftComplex** equalized_subframe_h)
{
	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	//Device data
	cufftComplex * subframe_d, *channel_d, *equalized_subframe_d;


	//Host data allocation
	*equalized_subframe_h = (cufftComplex *)malloc(sizeof(cufftComplex)*(M_pusch_sc*N_data_symbs_per_subframe));


	//Device data allocation
	startTimer();
	cudaMalloc((void **)&subframe_d, sizeof(cufftComplex)*M_pusch_sc*N_data_symbs_per_subframe);
	cudaMalloc((void **)&channel_d, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&equalized_subframe_d, sizeof(cufftComplex)*(M_pusch_sc*N_data_symbs_per_subframe));
	stopTimer("cudaMalloc Time= %.6f ms\n", elapsed);

	//Copying data to device
	startTimer();
	cudaMemcpy(subframe_d, subframe_h, sizeof(cufftComplex)*M_pusch_sc*N_data_symbs_per_subframe, cudaMemcpyHostToDevice);
	cudaMemcpy(channel_d, channel_h, sizeof(cufftComplex)*M_pusch_sc, cudaMemcpyHostToDevice);
	stopTimer("cudaMemcpy Host->Device Time= %.6f ms\n", elapsed);


	//Calling the kernel(s)

	
	dim3 grid(2, N_data_symbs_per_subframe, 1);
	dim3 block(M_pusch_sc / 2, 1, 1);
	startTimer();
	channel_equalization_zf << < grid, block >> > (subframe_d, channel_d, equalized_subframe_d, M_pusch_sc);
	stopTimer("channel equalization= %.6f ms\n", elapsed);



	cudaDeviceSynchronize();


	//Retrieve data from device
	startTimer();
	cudaMemcpy(*equalized_subframe_h, equalized_subframe_d, sizeof(cufftComplex)*M_pusch_sc*N_data_symbs_per_subframe, cudaMemcpyDeviceToHost);
	stopTimer("cudaMemcpy Device->Host Time= %.6f ms\n", elapsed);


	// Cleanup
	cudaFree(subframe_d);
	cudaFree(channel_d);
	cudaFree(equalized_subframe_d);


	//Destroy timers
	destroyTimers();

}
