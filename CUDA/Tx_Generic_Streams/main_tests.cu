#include "input.cuh"
#include "generate_psuedo_random_seq.cuh"
#include "interleaver.cuh"
#include "scrambler.cuh"
#include "mapper.cuh"
#include "transform_precoder.cuh"
#include "generate_dmrs_pusch.cuh"
#include "generate_ul_rs.cuh"
#include "compose_subframe.cuh"
#include "sc_fdma_modulator.cuh"
#include "main.cuh"

int main(int argc, char **argv) {

	//For timing purpose
	timerInit();
	startTimer();

	int N_bits, N_ri;
	const int Qm = 6;					// 64QAM Modulation
	const int N_l = 4;					// Number of Layers

	// Physical layer cell identity (we need for generation of random sequence)
	int N_id_cell = 2;						// assume enodeB scheduled cell 2 for the UE
	int M_pusch_rb = 100;					// number of resource blocks assigned to the UE
	int n_s = 0;							// assume UE send on time slot 4
	int n_RNTI = 10;						// radio network temporary identifier given to the UE by enodeB(assume 10)
	int N_subfr = 0;						// Subframe number within a radio frame
	BYTE* inputBits_h = readBits(argc, argv[1], &N_bits);			//Get input bits from the text file
	BYTE* riBits_h = readBits(argc, argv[2], &N_ri);					//Get RI bits from the text file

	//cudaMalloc & cudaMemcpy for inputBits & RI_Bits to Device
	Byte *inputBits_d = 0, *riBits_d = 0;

	cudaMalloc((void **)&inputBits_d, sizeof(Byte)*N_bits);
	cudaMalloc((void **)&riBits_d, sizeof(Byte)*N_ri);
	Byte* c_d = 0;
	cudaMalloc((void **)&c_d, sizeof(Byte)*N_bits);

	cudaMemcpyAsync(inputBits_d, inputBits_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(riBits_d, riBits_h, sizeof(Byte)*N_ri, cudaMemcpyHostToDevice);
	stopTimer("cudaMalloc & cudaMemcpy for inputBits & RI_Bits Time= %.6f ms\n", elapsed);

	//Create Plans
	startTimer();
	cufftHandle plan_transform_precoder_1;
	cufftHandle plan_transform_precoder_2;
	cufftHandle plan_transform_precoder_3;
	cufftHandle plan_transform_precoder_4;
	int n[1] = { N_sc_rb*M_pusch_rb };

	cufftPlanMany(&plan_transform_precoder_1, 1, n, NULL, 1, n[0], NULL, 1, N_sc_rb*M_pusch_rb, CUFFT_C2C, (((N_bits / N_l) + N_ri) / Qm) / n[0]);
	cufftPlanMany(&plan_transform_precoder_2, 1, n, NULL, 1, n[0], NULL, 1, N_sc_rb*M_pusch_rb, CUFFT_C2C, (((N_bits / N_l) + N_ri) / Qm) / n[0]);
	cufftPlanMany(&plan_transform_precoder_3, 1, n, NULL, 1, n[0], NULL, 1, N_sc_rb*M_pusch_rb, CUFFT_C2C, (((N_bits / N_l) + N_ri) / Qm) / n[0]);
	cufftPlanMany(&plan_transform_precoder_4, 1, n, NULL, 1, n[0], NULL, 1, N_sc_rb*M_pusch_rb, CUFFT_C2C, (((N_bits / N_l) + N_ri) / Qm) / n[0]);

	cufftHandle plan_sc_fdma_1;
	cufftHandle plan_sc_fdma_2;
	cufftHandle plan_sc_fdma_3;
	cufftHandle plan_sc_fdma_4;
	n[0] = { FFT_size };
	cufftPlanMany(&plan_sc_fdma_1, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);
	cufftPlanMany(&plan_sc_fdma_2, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);
	cufftPlanMany(&plan_sc_fdma_3, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);
	cufftPlanMany(&plan_sc_fdma_4, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);

	stopTimer("Create Plans Time= %.6f ms\n", elapsed);

	//Device data allocation
	startTimer();

	int data_vec_len = Qm*N_l;
	int ri_vec_len = Qm*N_l;
	int N_data_bits = N_bits / data_vec_len;
	int N_ri_bits = N_ri / data_vec_len;
	int H_prime = N_data_bits;
	int H_vec_len = data_vec_len;
	int H_prime_total = H_prime + N_ri_bits;

	int R_mux = (H_prime_total*Qm*N_l) / N_pusch_symbs;
	int R_prime_mux = R_mux / (Qm*N_l);

	Byte *y_idx_d, *y_mat_d;
	Byte *interleaved_d_1;
	Byte *interleaved_d_2;
	Byte *interleaved_d_3;
	Byte *interleaved_d_4;
	//Note: If you create streams, y_idx_d & y_mat_d should be duplicated
	cudaMalloc((void **)&y_idx_d, sizeof(Byte)*(N_pusch_symbs * R_prime_mux));
	cudaMalloc((void **)&y_mat_d, sizeof(Byte)*(N_pusch_symbs*R_mux));
	cudaMalloc((void **)&interleaved_d_1, sizeof(Byte)*(N_pusch_symbs*R_mux / N_l));
	cudaMalloc((void **)&interleaved_d_2, sizeof(Byte)*(N_pusch_symbs*R_mux / N_l));
	cudaMalloc((void **)&interleaved_d_3, sizeof(Byte)*(N_pusch_symbs*R_mux / N_l));
	cudaMalloc((void **)&interleaved_d_4, sizeof(Byte)*(N_pusch_symbs*R_mux / N_l));

	Byte *scrambledbits_d_1 = 0;
	Byte *scrambledbits_d_2 = 0;
	Byte *scrambledbits_d_3 = 0;
	Byte *scrambledbits_d_4 = 0;
	cudaMalloc((void **)&scrambledbits_d_1, sizeof(Byte)*N_bits / N_l);
	cudaMalloc((void **)&scrambledbits_d_2, sizeof(Byte)*N_bits / N_l);
	cudaMalloc((void **)&scrambledbits_d_3, sizeof(Byte)*N_bits / N_l);
	cudaMalloc((void **)&scrambledbits_d_4, sizeof(Byte)*N_bits / N_l);

	Byte *bits_each_Qm_d_1;
	Byte *bits_each_Qm_d_2;
	Byte *bits_each_Qm_d_3;
	Byte *bits_each_Qm_d_4;
	float* symbols_R_d_1 = 0;
	float* symbols_I_d_1 = 0;
	float* symbols_R_d_2 = 0;
	float* symbols_I_d_2 = 0;
	float* symbols_R_d_3 = 0;
	float* symbols_I_d_3 = 0;
	float* symbols_R_d_4 = 0;
	float* symbols_I_d_4 = 0;

	cudaMalloc((void **)&bits_each_Qm_d_1, sizeof(Byte)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&bits_each_Qm_d_2, sizeof(Byte)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&bits_each_Qm_d_3, sizeof(Byte)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&bits_each_Qm_d_4, sizeof(Byte)*(N_bits / (Qm*N_l)));

	cudaMalloc((void **)&symbols_R_d_1, sizeof(float)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&symbols_I_d_1, sizeof(float)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&symbols_R_d_2, sizeof(float)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&symbols_I_d_2, sizeof(float)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&symbols_R_d_3, sizeof(float)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&symbols_I_d_3, sizeof(float)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&symbols_R_d_4, sizeof(float)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&symbols_I_d_4, sizeof(float)*(N_bits / (Qm*N_l)));

	cufftComplex *precoded_symbols_d_1 = 0;
	cufftComplex *precoded_symbols_d_2 = 0;
	cufftComplex *precoded_symbols_d_3 = 0;
	cufftComplex *precoded_symbols_d_4 = 0;
	cufftComplex *cuComplex_symbols_d_1 = 0;
	cufftComplex *cuComplex_symbols_d_2 = 0;
	cufftComplex *cuComplex_symbols_d_3 = 0;
	cufftComplex *cuComplex_symbols_d_4 = 0;

	cudaMalloc((void **)&cuComplex_symbols_d_1, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&cuComplex_symbols_d_2, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&cuComplex_symbols_d_3, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&cuComplex_symbols_d_4, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));

	cudaMalloc((void **)&precoded_symbols_d_1, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&precoded_symbols_d_2, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&precoded_symbols_d_3, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
	cudaMalloc((void **)&precoded_symbols_d_4, sizeof(cufftComplex)*(N_bits / (Qm*N_l)));

	cufftComplex* x_q_d_1;
	cufftComplex *x_q_d_2;
	cufftComplex* x_q_d_3;
	cufftComplex *x_q_d_4;
	cufftComplex* dmrs_1_d_1 = 0;
	cufftComplex *dmrs_1_d_2 = 0;
	cufftComplex *dmrs_2_d_1 = 0;
	cufftComplex *dmrs_2_d_2 = 0;
	cufftComplex* dmrs_3_d_1 = 0;
	cufftComplex *dmrs_3_d_2 = 0;
	cufftComplex *dmrs_4_d_1 = 0;
	cufftComplex *dmrs_4_d_2 = 0;

	cudaMalloc((void **)&dmrs_1_d_1, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_1_d_2, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_2_d_1, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_2_d_2, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_3_d_1, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_3_d_2, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_4_d_1, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_4_d_2, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&x_q_d_1, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);
	cudaMalloc((void **)&x_q_d_2, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);
	cudaMalloc((void **)&x_q_d_3, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);
	cudaMalloc((void **)&x_q_d_4, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);

	cufftComplex *subframe_d_1 = 0;
	cufftComplex *subframe_d_2 = 0;
	cufftComplex *subframe_d_3 = 0;
	cufftComplex *subframe_d_4 = 0;

	cudaMalloc((void **)&subframe_d_1, sizeof(cufftComplex)*N_symbs_per_subframe*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&subframe_d_2, sizeof(cufftComplex)*N_symbs_per_subframe*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&subframe_d_3, sizeof(cufftComplex)*N_symbs_per_subframe*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&subframe_d_4, sizeof(cufftComplex)*N_symbs_per_subframe*N_sc_rb*M_pusch_rb);

	cufftComplex* ifft_vec_d_1;
	cufftComplex *ifft_vec_d_2;
	cufftComplex* ifft_vec_d_3;
	cufftComplex *ifft_vec_d_4;

	cufftComplex *pusch_bb_d_1 = 0;
	cufftComplex *pusch_bb_d_2 = 0;
	cufftComplex *pusch_bb_d_3 = 0;
	cufftComplex *pusch_bb_d_4 = 0;

	cudaMalloc((void **)&ifft_vec_d_1, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&ifft_vec_d_2, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&ifft_vec_d_3, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&ifft_vec_d_4, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&pusch_bb_d_1, sizeof(cufftComplex)*modulated_subframe_length);
	cudaMalloc((void **)&pusch_bb_d_2, sizeof(cufftComplex)*modulated_subframe_length);
	cudaMalloc((void **)&pusch_bb_d_3, sizeof(cufftComplex)*modulated_subframe_length);
	cudaMalloc((void **)&pusch_bb_d_4, sizeof(cufftComplex)*modulated_subframe_length);

	stopTimer("Device data allocation Time= %.6f ms\n", elapsed);

	startTimer();

	//Generate Pseudo Random Seq.
	Byte *c_h = 0;
	generate_psuedo_random_seq(&c_h, N_bits, n_RNTI, n_s, N_id_cell);

	//Copy (c) to Device
	cudaMemcpyAsync(c_d, c_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);

	//Interleaver
	//Byte** interleaved_set[4] = { &interleaved_d_1 ,
	//	&interleaved_d_2 ,
	//	&interleaved_d_3 ,
	//	&interleaved_d_4 };
	//Interleaver will be modified from inside in higher order of MIMO
	interleaver(inputBits_d, riBits_d, &interleaved_d_1, &interleaved_d_2, &interleaved_d_3, &interleaved_d_4, N_bits, N_ri, Qm, N_l, y_idx_d, y_mat_d);

	//Scrambler
	scrambler(interleaved_d_1, &scrambledbits_d_1, c_d, (N_bits / N_l) + N_ri);
	scrambler(interleaved_d_2, &scrambledbits_d_2, c_d, (N_bits / N_l) + N_ri);
	scrambler(interleaved_d_3, &scrambledbits_d_3, c_d, (N_bits / N_l) + N_ri);
	scrambler(interleaved_d_4, &scrambledbits_d_4, c_d, (N_bits / N_l) + N_ri);

	//Mapper
	mapper(scrambledbits_d_1, (N_bits / N_l) + N_ri, Qm, &symbols_R_d_1, &symbols_I_d_1, bits_each_Qm_d_1);
	mapper(scrambledbits_d_2, (N_bits / N_l) + N_ri, Qm, &symbols_R_d_2, &symbols_I_d_2, bits_each_Qm_d_2);
	mapper(scrambledbits_d_3, (N_bits / N_l) + N_ri, Qm, &symbols_R_d_3, &symbols_I_d_3, bits_each_Qm_d_3);
	mapper(scrambledbits_d_4, (N_bits / N_l) + N_ri, Qm, &symbols_R_d_4, &symbols_I_d_4, bits_each_Qm_d_4);

	//float* hprint = (float *)malloc(sizeof(float)*(14400));
	//cudaMemcpy(hprint, symbols_I_d_4, sizeof(float)*(14400), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < 14400; i++)
	//{
	//	printf("%10f", hprint[i]);
	//}

	
	//Transform Precoder
	transform_precoder(symbols_R_d_1, symbols_I_d_1, M_pusch_rb, ((N_bits / N_l) + N_ri) / Qm, &precoded_symbols_d_1, plan_transform_precoder_1, cuComplex_symbols_d_1);
	transform_precoder(symbols_R_d_2, symbols_I_d_2, M_pusch_rb, ((N_bits / N_l) + N_ri) / Qm, &precoded_symbols_d_2, plan_transform_precoder_2, cuComplex_symbols_d_2);
	transform_precoder(symbols_R_d_3, symbols_I_d_3, M_pusch_rb, ((N_bits / N_l) + N_ri) / Qm, &precoded_symbols_d_3, plan_transform_precoder_3, cuComplex_symbols_d_3);
	transform_precoder(symbols_R_d_4, symbols_I_d_4, M_pusch_rb, ((N_bits / N_l) + N_ri) / Qm, &precoded_symbols_d_4, plan_transform_precoder_4, cuComplex_symbols_d_4);

	//long NZ = 100;
	//cufftComplex* hprint = (cufftComplex *)malloc(sizeof(cufftComplex)*(NZ));
	//cudaMemcpy(hprint, precoded_symbols_d_4, sizeof(cufftComplex)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%10f", hprint[i].x);
	//}

	//Generate DMRS
	generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 0, &dmrs_1_d_1, &dmrs_1_d_2, x_q_d_1);
	generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 1, &dmrs_2_d_1, &dmrs_2_d_2, x_q_d_2);
	generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 2, &dmrs_3_d_1, &dmrs_3_d_2, x_q_d_3);
	generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 3, &dmrs_4_d_1, &dmrs_4_d_2, x_q_d_4);
	//update on github

	//long NZ = 100;
	//cufftComplex* hprint = (cufftComplex *)malloc(sizeof(cufftComplex)*(NZ));
	//cudaMemcpy(hprint, dmrs_4_d_2, sizeof(cufftComplex)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%10f", hprint[i].x);
	//}

	//Multiplexing the DMRS with the Data
	compose_subframe(precoded_symbols_d_1, dmrs_1_d_1, dmrs_1_d_2, M_pusch_rb, &subframe_d_1);
	compose_subframe(precoded_symbols_d_2, dmrs_2_d_1, dmrs_2_d_2, M_pusch_rb, &subframe_d_2);
	compose_subframe(precoded_symbols_d_3, dmrs_3_d_1, dmrs_3_d_2, M_pusch_rb, &subframe_d_3);
	compose_subframe(precoded_symbols_d_4, dmrs_4_d_1, dmrs_4_d_2, M_pusch_rb, &subframe_d_4);

	//long NZ = 100;
	//cufftComplex* hprint = (cufftComplex *)malloc(sizeof(cufftComplex)*(NZ));
	//cudaMemcpy(hprint, subframe_d_4, sizeof(cufftComplex)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%10f", hprint[i].x);
	//}

	// Generate SC-FDMA signal
	sc_fdma_modulator(subframe_d_1, M_pusch_rb, &pusch_bb_d_1, plan_sc_fdma_1, ifft_vec_d_1);
	sc_fdma_modulator(subframe_d_2, M_pusch_rb, &pusch_bb_d_2, plan_sc_fdma_2, ifft_vec_d_2);
	sc_fdma_modulator(subframe_d_3, M_pusch_rb, &pusch_bb_d_3, plan_sc_fdma_3, ifft_vec_d_3);
	sc_fdma_modulator(subframe_d_4, M_pusch_rb, &pusch_bb_d_4, plan_sc_fdma_4, ifft_vec_d_4);

	cufftComplex *pusch_bb_h_1 = (cufftComplex *)malloc(sizeof(cufftComplex)*(modulated_subframe_length));
	cufftComplex *pusch_bb_h_2 = (cufftComplex *)malloc(sizeof(cufftComplex)*(modulated_subframe_length));
	cufftComplex *pusch_bb_h_3 = (cufftComplex *)malloc(sizeof(cufftComplex)*(modulated_subframe_length));
	cufftComplex *pusch_bb_h_4 = (cufftComplex *)malloc(sizeof(cufftComplex)*(modulated_subframe_length));
	cudaMemcpy(pusch_bb_h_1, pusch_bb_d_1, sizeof(cufftComplex)*(modulated_subframe_length), cudaMemcpyDeviceToHost);
	cudaMemcpy(pusch_bb_h_2, pusch_bb_d_2, sizeof(cufftComplex)*(modulated_subframe_length), cudaMemcpyDeviceToHost);
	cudaMemcpy(pusch_bb_h_3, pusch_bb_d_3, sizeof(cufftComplex)*(modulated_subframe_length), cudaMemcpyDeviceToHost);
	cudaMemcpy(pusch_bb_h_4, pusch_bb_d_4, sizeof(cufftComplex)*(modulated_subframe_length), cudaMemcpyDeviceToHost);

	stopTimer("Processing Time= %.6f ms\n", elapsed);

	//To compare with MATLAB results
	//Run the file (output.m)
	int NNN = modulated_subframe_length;
	FILE *results;
	if ((results = freopen("output.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	//1st Layer
	printf("clear; clc;\nsymbols_real = [ ");
	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_1[i].x);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\nsymbols_imag = [ ");

	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_1[i].y);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_CUDA_1 = symbols_real + 1i * symbols_imag;\n");

	//2nd Layer
	printf("\nsymbols_real = [ ");
	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_2[i].x);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\nsymbols_imag = [ ");

	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_2[i].y);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_CUDA_2 = symbols_real + 1i * symbols_imag;\n");

	//3rd Layer
	printf("\nsymbols_real = [ ");
	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_3[i].x);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\nsymbols_imag = [ ");

	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_3[i].y);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_CUDA_3 = symbols_real + 1i * symbols_imag;\n");

	//4th Layer
	printf("\nsymbols_real = [ ");
	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_4[i].x);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\nsymbols_imag = [ ");

	for (int i = 0; i < NNN; i++)
	{
		printf("%10f", pusch_bb_h_4[i].y);
		if (i != (NNN - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_CUDA_4 = symbols_real + 1i * symbols_imag;\n");

	//Matlab code
	printf("matlab_test");

	fclose(results);

	if ((results = freopen("matlab_test.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	printf("N_bits = %d; \n", N_bits);
	if (Qm == 6)
		printf("mod_type = %s; \n", "'64qam'");
	else if (Qm == 4)
		printf("mod_type = %s; \n", "'16qam'");
	else if (Qm == 2)
		printf("mod_type = %s; \n", "'qpsk'");
	else if (Qm == 1)
		printf("mod_type = %s; \n", "'bpsk'");

	printf("N_sc_rb   = 12;      %% number of subcarriers in each resource block\n");
	printf("M_pusch_rb = %d;      %% number of resource blocks assigned to the UE\n", M_pusch_rb);
	printf("M_pusch_sc = M_pusch_rb*N_sc_rb;  %% total number of subcarriers\n\n");
	printf("N_l = %d; \nQ_m = %d; \ndata_bits_total = (fread(fopen('%s')) - '0').';\ndata_bits_1 = data_bits_total(1:N_bits/2);\ndata_bits_2 = data_bits_total( (N_bits/2) + 1 : end);\nri_bits = (fread(fopen('%s'))-'0').';\n", N_l, Qm, argv[1], argv[argc - 1]);
	printf("interleaved_bits = channel_interleaver_MIMO([data_bits_1; data_bits_2].', ri_bits, [], N_l, Q_m);\ninterleaved_bits_2_col = reshape(interleaved_bits,length(interleaved_bits)/2,2);\nc_init = 10 * 2 ^ 14 + floor(0 / 2) * 2 ^ 9 + 2; \nc = generate_psuedo_random_seq(c_init, length(interleaved_bits_2_col)); \nscrambled = scrambler_MIMO(interleaved_bits_2_col.', [c; c], N_l); \nmodulated_symbols1 = mapper(scrambled(1,:), mod_type);\nmodulated_symbols2 = mapper(scrambled(2,:), mod_type);\nX = layer_mapping(modulated_symbols1, modulated_symbols2, 0, 0, N_l);\ntransform_precoded_symbols = transform_precoder_mimo(X, M_pusch_rb, N_l); \nprecoded_symbols = precoding_mimo(transform_precoded_symbols,N_l,N_l);\ndmrs = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 0);\ndmrs_1_1 = dmrs(1:M_pusch_sc);\ndmrs_1_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);\ndmrs = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 1);\ndmrs_2_1 = dmrs(1:M_pusch_sc);\ndmrs_2_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);\ndmrs1 = [dmrs_1_1; dmrs_1_2];\ndmrs2 = [dmrs_2_1; dmrs_2_2];\nsubframe_per_ant = compose_subframe_mimo(precoded_symbols, dmrs1, dmrs2, M_pusch_rb, N_l);\nsymbols_MATLAB = sc_fdma_modulator_MIMO(subframe_per_ant, M_pusch_rb, N_l);\nsymbols_MATLAB = reshape(symbols_MATLAB,1,length(symbols_MATLAB)*N_l);");
	printf("\nsymbols_CUDA = [symbols_CUDA_1 symbols_CUDA_2 symbols_CUDA_3 symbols_CUDA_4];\n\nsum((abs(symbols_MATLAB) - abs(symbols_CUDA)))");

	fclose(results);
	
	}