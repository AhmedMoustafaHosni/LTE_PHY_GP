//  Function:    channe_estimation
//	Description : Generates channel estimation
//	Inputs : symb_0 - received DMRS number 1
//			 symb_1 - received DMRS number 2
//			 dmrs_0 - generated DMRS number 1
//			 dmrs_1 - generated DMRS number 2
//			 M_pusch_sc - number of subcarriers allocated to ue
//	Outputs : channel - channel estimation matrix to be used for equalization
//			  Noise_power : N0
//	edit : 9 / 3 / 2017
//	By : Ahmed Moustafa

#include "channel_estimation.cuh"



__global__ void divide_pilots(cufftComplex* symb_0_d, cufftComplex* symb_1_d, cufftComplex* dmrs_0_d, cufftComplex* dmrs_1_d, cufftComplex* filtered_pilots1_d, cufftComplex* filtered_pilots2_d, const int M_pusch_sc)
{
 	int x_idx = blockIdx.x;
	int y_idx = threadIdx.y + blockIdx.y * blockDim.y ;

	if (y_idx >= M_pusch_sc)
		return;

	// divide the received pilots over the generated reference to get value of the channel 
	if (x_idx == 0)
	{
		filtered_pilots1_d[y_idx] = cuCdivf(symb_0_d[y_idx], dmrs_0_d[y_idx]);
	}
	else
	{
		filtered_pilots2_d[y_idx] = cuCdivf(symb_1_d[y_idx], dmrs_1_d[y_idx]);
	}


	// sync the threads to go  the time average 
	__syncthreads();
	
	if (x_idx != 0)
		return;


	//time average 
	filtered_pilots1_d[y_idx] = (filtered_pilots1_d[y_idx] + filtered_pilots2_d[y_idx] ) / 2;

	// sync the threads for freq average
	__syncthreads();


	// freq avergae with moving average filter of window size 19
	if (y_idx > 8 && y_idx < (M_pusch_sc - 9) )

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 9] + filtered_pilots1_d[y_idx - 8] + filtered_pilots1_d[y_idx - 7] + filtered_pilots1_d[y_idx - 6] + filtered_pilots1_d[y_idx - 5] + filtered_pilots1_d[y_idx - 4] + filtered_pilots1_d[y_idx - 3] + filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2] + filtered_pilots1_d[y_idx + 3] + filtered_pilots1_d[y_idx + 4] + filtered_pilots1_d[y_idx + 5] + filtered_pilots1_d[y_idx + 6] + filtered_pilots1_d[y_idx + 7] + filtered_pilots1_d[y_idx + 8] + filtered_pilots1_d[y_idx + 9]) / 19;

	else if (y_idx == 0 || y_idx == (M_pusch_sc - 1))

		filtered_pilots2_d[y_idx] = filtered_pilots1_d[y_idx];

	else if(y_idx == 1 || y_idx == (M_pusch_sc - 2))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx-1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx+1]) / 3;

	else if(y_idx == 2 || y_idx == (M_pusch_sc - 3))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2]) / 5;

	else if(y_idx == 3 || y_idx == (M_pusch_sc - 4))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 3] + filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2] + filtered_pilots1_d[y_idx + 3]) / 7;

	else if(y_idx == 4 || y_idx == (M_pusch_sc - 5))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 4] + filtered_pilots1_d[y_idx - 3] + filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2] + filtered_pilots1_d[y_idx + 3] + filtered_pilots1_d[y_idx + 4] ) / 9;

	else if(y_idx == 5 || y_idx == (M_pusch_sc - 6))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 5] + filtered_pilots1_d[y_idx - 4] + filtered_pilots1_d[y_idx - 3] + filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2] + filtered_pilots1_d[y_idx + 3] + filtered_pilots1_d[y_idx + 4] + filtered_pilots1_d[y_idx + 5]) / 11;

	else if(y_idx == 6 || y_idx == (M_pusch_sc - 7))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 6] + filtered_pilots1_d[y_idx - 5] + filtered_pilots1_d[y_idx - 4] + filtered_pilots1_d[y_idx - 3] + filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2] + filtered_pilots1_d[y_idx + 3] + filtered_pilots1_d[y_idx + 4] + filtered_pilots1_d[y_idx + 5] + filtered_pilots1_d[y_idx + 6]) / 13;

	else if(y_idx == 7 || y_idx == (M_pusch_sc - 8))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 7] + filtered_pilots1_d[y_idx - 6] + filtered_pilots1_d[y_idx - 5] + filtered_pilots1_d[y_idx - 4] + filtered_pilots1_d[y_idx - 3] + filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2] + filtered_pilots1_d[y_idx + 3] + filtered_pilots1_d[y_idx + 4] + filtered_pilots1_d[y_idx + 5] + filtered_pilots1_d[y_idx + 6] + filtered_pilots1_d[y_idx + 7]) / 15;

	else if(y_idx == 8 || y_idx == (M_pusch_sc - 9))

		filtered_pilots2_d[y_idx] = (filtered_pilots1_d[y_idx - 8] + filtered_pilots1_d[y_idx - 7] + filtered_pilots1_d[y_idx - 6] + filtered_pilots1_d[y_idx - 5] + filtered_pilots1_d[y_idx - 4] + filtered_pilots1_d[y_idx - 3] + filtered_pilots1_d[y_idx - 2] + filtered_pilots1_d[y_idx - 1] + filtered_pilots1_d[y_idx] + filtered_pilots1_d[y_idx + 1] + filtered_pilots1_d[y_idx + 2] + filtered_pilots1_d[y_idx + 3] + filtered_pilots1_d[y_idx + 4] + filtered_pilots1_d[y_idx + 5] + filtered_pilots1_d[y_idx + 6] + filtered_pilots1_d[y_idx + 7] + filtered_pilots1_d[y_idx + 8]) / 17;


	//for mimo

	//// sync the threads for noise estimation
	//__syncthreads();


	////noise estimate
	//filtered_pilots1_d[y_idx] = filtered_pilots1_d[y_idx] - filtered_pilots2_d[y_idx];



}



//void channe_estimation(cufftComplex* symb_0_h, cufftComplex* symb_1_h, cufftComplex* dmrs_0_h, cufftComplex* dmrs_1_h, const int M_pusch_sc, cufftComplex** channel_h)
void channe_estimation(cufftComplex* symb_0_d, cufftComplex* symb_1_d, cufftComplex* dmrs_0_d, cufftComplex* dmrs_1_d, const int M_pusch_sc, cufftComplex** channel_d)
{
	cufftComplex* filtered_pilots1_d, *filtered_pilots2_d, *filtered_pilots2_h;
	
	cudaMalloc((void **)&filtered_pilots1_d, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&filtered_pilots2_d, sizeof(cufftComplex)*M_pusch_sc);
	
	filtered_pilots2_h = (cufftComplex*) malloc(sizeof(cufftComplex)*M_pusch_sc);

	//Calling the kernel(s)

	//---------------------------------------step 1 : Determine channel estimates
	int thread_y, grid_y;
	if (M_pusch_sc > 1024)
	{
		thread_y = M_pusch_sc/2;
		grid_y = 2;
	}
	else
	{
		thread_y = M_pusch_sc;
		grid_y = 1;
	}
	dim3 blockDim(1, thread_y);
	dim3 gridDim(2, grid_y);
	
	//startTimer();
	divide_pilots << < gridDim, blockDim >> > (symb_0_d, symb_1_d, dmrs_0_d, dmrs_1_d, filtered_pilots1_d, filtered_pilots2_d, M_pusch_sc);
	//stopTimer("divide_pilots= %.6f ms\n", elapsed);
	
	*channel_d = filtered_pilots2_d;
	//cudaMemcpy(filtered_pilots2_h, filtered_pilots2_d, sizeof(cufftComplex)*M_pusch_sc, cudaMemcpyDeviceToHost);
	//FILE *results1;
	//if ((results1 = freopen("channel.m", "w+", stdout)) == NULL) {
	//	printf("Cannot open file.\n");
	//	//exit(1);
	//}

	////input subframe
	//printf("clc;\nx_inside = [ ");
	//for (int i = 0; i < M_pusch_sc; i++)
	//{
	//	printf("%f ", filtered_pilots2_h[i].x);
	//}
	//printf(" ]; ");
	//printf("\n");
	//printf("y_inside = [ ");
	//for (int i = 0; i < M_pusch_sc; i++)
	//{
	//	printf("%f ", filtered_pilots2_h[i].y);
	//}
	//printf(" ];\n ");
	//printf("channel_cuda_inside = x_inside + 1i * y_inside;\nsum(round(channel_cuda_inside,6) - round(channel_cuda,6))");
	//fclose(results1);

	// Cleanup
	/*cudaFree(symb_0_d);
	cudaFree(symb_1_d);
	cudaFree(dmrs_0_d);
	cudaFree(dmrs_1_d);
	*///cudaFree(filtered_pilots1_d);
	//cudaFree(filtered_pilots2_d);
	//cudaFree(channel_d);

	//Destroy timers
	//destroyTimers();

}
