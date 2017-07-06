/*
% Function:		sc_fdma_demodulator
% Description:	Generates complex symbols from the sc-fdma symbols
% Inputs		*pusch_bb_h		sc-fdma symbols
M_pusch_rb		numer of resource blocks assigned to ue
% Outputs:		*symbs_h		output symbols
By: Mohammed Mostafa
*/


#include "sc_fdma_demodulator.cuh"

__global__ void construct_fft_vec(cufftComplex* pusch_bb_d, cufftComplex* fft_vec_d, int M_pusch_sc) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y;
	
	if (y_idx == 0) //160 = N_cp_L_0
		fft_vec_d[x_idx] = pusch_bb_d[x_idx + 160]; 
	else if (y_idx == 7)// 14336 = y_idx * FFT_size   // 15520 =  y_idx * FFT_size + 2*N_cp_L_0 + 6*N_cp_L_else
		fft_vec_d[14336 + x_idx] = pusch_bb_d[x_idx + 15520]; 
	else if (y_idx < 7)
		fft_vec_d[y_idx * FFT_size + x_idx] = pusch_bb_d[x_idx + y_idx*FFT_size + N_cp_L_0 + y_idx*N_cp_L_else];
	else
		fft_vec_d[y_idx * FFT_size + x_idx] = pusch_bb_d[x_idx + y_idx*FFT_size + 176 + y_idx*N_cp_L_else]; // 176 = 2*N_cp_L_0 - N_cp_L_else
}

__global__ void extract_symbs(cufftComplex* fft_vec_d, cufftComplex* symbs_d, int M_pusch_sc_div2) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y;
	
	//if (x_idx >= M_pusch_sc)
		//return;
	symbs_d[y_idx*M_pusch_sc_div2*2 + x_idx] = fft_vec_d[y_idx*FFT_size + (x_idx + FFT_size - M_pusch_sc_div2)%FFT_size]; // 1448 = FFT_size - M_pusch_sc/2 
}

void sc_fdma_demodulator(cufftComplex* pusch_bb_d, const int M_pusch_rb, cufftComplex** symbs_d, cufftHandle plan_sc_fdma, cufftComplex* fft_vec_d)
{
	int M_pusch_sc = N_sc_rb * M_pusch_rb;
	
	//constructing fft_vec
	dim3 grid(2, N_symbs_per_subframe,1);
	dim3 block(1024,1,1);
	construct_fft_vec <<< grid, block >>>(pusch_bb_d, fft_vec_d, M_pusch_sc);

	// CUFFT plan
	cufftExecC2C(plan_sc_fdma, fft_vec_d, fft_vec_d, CUFFT_FORWARD);

	dim3 grid1(2, N_symbs_per_subframe, 1);
	dim3 block1(M_pusch_sc/2, 1, 1);
	extract_symbs << < grid1, block1 >> >(fft_vec_d, *symbs_d, M_pusch_sc/2);
	

	//Cleanup
	//cudaFree(pusch_bb_d);
	//cudaFree(fft_vec_d);
	//cudaFree(symbs_d);

}
