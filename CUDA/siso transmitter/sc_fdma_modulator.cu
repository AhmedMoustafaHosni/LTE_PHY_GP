/*
% Function:	sc_fdma_modulator
% Description:	Generates sc-fdma signal of the subframe
% Inputs:	*subframe_h:	received DMRS number 1
% 		M_pusch_rb	numer of resource blocks assigned to ue
% Outputs:	*pusch_bb_h	base band signal
By: Ahmad Nour & Mohammed Mostafa
*/


#include "sc_fdma_modulator.cuh"

__global__ void reshape_ifft_vec(cufftComplex* subframe_d, cufftComplex* ifft_vec_d, int M_pusch_sc_div2) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx_ifft = blockIdx.y * FFT_size;
	int y_idx_subframe = blockIdx.y * M_pusch_sc_div2 * 2;

	if (x_idx < M_pusch_sc_div2) //M_pusch_sc / 2
		ifft_vec_d[y_idx_ifft + x_idx] = subframe_d[y_idx_subframe + x_idx + M_pusch_sc_div2] / FFT_size;    // 600 = M_pusch_sc / 2
	else if (x_idx >= FFT_size - M_pusch_sc_div2) //FFT_size - M_pusch_sc / 2
		ifft_vec_d[y_idx_ifft + x_idx] = subframe_d[y_idx_subframe + x_idx - FFT_size + M_pusch_sc_div2] / FFT_size;   //1448 = FFT_size - M_pusch_sc / 2
	else
	{
		ifft_vec_d[y_idx_ifft + x_idx].x = 0;
		ifft_vec_d[y_idx_ifft + x_idx].y = 0;
	}
		
}

__global__ void add_cyclic_prefix(cufftComplex* ifft_vec_d, cufftComplex* pusch_bb_d, int M_pusch_sc) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y;

	//if (x_idx > 2207)  //2191 = FFT_size + N_cp_L_else - 1 = 2048 + 144 - 1
		//return;

	if (y_idx != 0 && y_idx != 7 && x_idx > 2191)  //2191 = FFT_size + N_cp_L_else - 1 = 2048 + 144 - 1
		return;

	if (y_idx == 0)
	{
		pusch_bb_d[x_idx] = ifft_vec_d[ (x_idx + FFT_size - N_cp_L_0)%FFT_size];
	}
	else if (y_idx == 7) // 15360 = FFT_size*7 + 144*6 + 160     // 14336 = y_idx*FFT_size = 7*2048
	{
		pusch_bb_d[15360 + x_idx] = ifft_vec_d[14336 + (x_idx + FFT_size - N_cp_L_0) % FFT_size];
	}
	else if (y_idx < 7)
	{
		pusch_bb_d[y_idx*FFT_size + N_cp_L_else*(y_idx-1)+ N_cp_L_0 + x_idx] = ifft_vec_d[y_idx*FFT_size + (x_idx + FFT_size - N_cp_L_else) % FFT_size];
	}
	else   //320 = 2*N_cp_L_0
	{
		pusch_bb_d[y_idx*FFT_size + N_cp_L_else*(y_idx - 2) + 320 + x_idx] = ifft_vec_d[y_idx*FFT_size + (x_idx + FFT_size - N_cp_L_else) % FFT_size];
	}
}

void sc_fdma_modulator(cufftComplex* subframe_d, const int M_pusch_rb, cufftComplex** pusch_bb_d, cufftHandle plan_sc_fdma, cufftComplex* ifft_vec_d)
{
	int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//constructing fft_vec
	dim3 grid(2, N_symbs_per_subframe,1);
	dim3 block(1024,1,1);
	reshape_ifft_vec <<< grid, block >>>(subframe_d, ifft_vec_d, M_pusch_sc/2);

	// CUFFT plan
	//int N_SIGS = N_symbs_per_subframe;
	//int n[1] = { FFT_size };
	//cufftHandle plan;
	//cufftPlanMany(&plan, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_SIGS);

	cufftExecC2C(plan_sc_fdma, ifft_vec_d, ifft_vec_d, CUFFT_INVERSE);

	dim3 grid1(3, N_symbs_per_subframe, 1);
	dim3 block1(736, 1, 1);						//(14*2048+14*160)/14*3 = 736      //(2048+160)/3 = 736
	add_cyclic_prefix << < grid1, block1 >> >(ifft_vec_d, *pusch_bb_d, M_pusch_sc);
	
}
