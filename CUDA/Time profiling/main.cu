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
	const int N_l = 1;					// Number of Layers

	// Physical layer cell identity (we need for generation of random sequence)
	int N_id_cell = 2;						// assume enodeB scheduled cell 2 for the UE
	int M_pusch_rb = 6;					// number of resource blocks assigned to the UE
	int n_s = 0;							// assume UE send on time slot 4
	int n_RNTI = 10;						// radio network temporary identifier given to the UE by enodeB(assume 10)
	int N_subfr = 0;						// Subframe number within a radio frame
	BYTE* inputBits_h = readBits(argc, argv[1], &N_bits);			//Get input bits from the text file
	BYTE* riBits_h = readBits(argc, argv[2], &N_ri);					//Get RI bits from the text file

	//Copy inputBits & RI_Bits to Device
	Byte *inputBits_d = 0, *riBits_d = 0;

	cudaMalloc((void **)&inputBits_d, sizeof(Byte)*N_bits);
	cudaMalloc((void **)&riBits_d, sizeof(Byte)*N_ri);
	Byte* c_d = 0;
	cudaMalloc((void **)&c_d, sizeof(Byte)*N_bits);

	cudaMemcpy(inputBits_d, inputBits_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);
	cudaMemcpy(riBits_d, riBits_h, sizeof(Byte)*N_ri, cudaMemcpyHostToDevice);
	stopTimer("Time= %.6f ms\n", elapsed);

	//Create Plans

	cufftHandle plan_transform_precoder;
	int n[1] = { N_sc_rb*M_pusch_rb };
	cufftPlanMany(&plan_transform_precoder, 1, n, NULL, 1, n[0], NULL, 1, N_sc_rb*M_pusch_rb, CUFFT_C2C, ((N_bits + N_ri) / Qm)/n[0]);

	cufftHandle plan_sc_fdma;
	n[0] = { FFT_size };
	cufftPlanMany(&plan_sc_fdma, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);

	//Allocation
	//startTimer();

	//timer_test << <1, 1 >> > ();

	int data_vec_len = Qm*N_l;
	int ri_vec_len = Qm*N_l;
	int N_data_bits = N_bits / data_vec_len;
	int N_ri_bits = N_ri / data_vec_len;
	int H_prime = N_data_bits;
	int H_vec_len = data_vec_len;
	int H_prime_total = H_prime + N_ri_bits;

	int R_mux = (H_prime_total*Qm*N_l) / N_pusch_symbs;
	int R_prime_mux = R_mux / (Qm*N_l);

	//Device data allocation
	Byte *y_idx_d, *y_mat_d, *interleaved_d;
	cudaMalloc((void **)&y_idx_d, sizeof(Byte)*(N_pusch_symbs * R_prime_mux));
	cudaMalloc((void **)&y_mat_d, sizeof(Byte)*(N_pusch_symbs*R_mux));
	cudaMalloc((void **)&interleaved_d, sizeof(Byte)*(N_pusch_symbs*R_mux));

	Byte *scrambledbits_d = 0;
	cudaMalloc((void **)&scrambledbits_d, sizeof(Byte)*N_bits);

	//Device data
	Byte *bits_each_Qm_d;
	float* symbols_R_d = 0, *symbols_I_d = 0;
	cudaMalloc((void **)&bits_each_Qm_d, sizeof(Byte)*(N_bits / Qm));
	cudaMalloc((void **)&symbols_R_d, sizeof(float)*(N_bits / Qm));
	cudaMalloc((void **)&symbols_I_d, sizeof(float)*(N_bits / Qm));

	cufftComplex *precoded_symbols_d = 0, *cuComplex_symbols_d = 0;
	cudaMalloc((void **)&cuComplex_symbols_d, sizeof(cufftComplex)*(N_bits / Qm));
	cudaMalloc((void **)&precoded_symbols_d, sizeof(cufftComplex)*(N_bits / Qm));

	cufftComplex* x_q_d;
	cufftComplex* dmrs_1_d = 0, *dmrs_2_d = 0;
	cudaMalloc((void **)&dmrs_1_d, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs_2_d, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&x_q_d, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);
	
	cufftComplex *subframe_d = 0;
	cudaMalloc((void **)&subframe_d, sizeof(cufftComplex)*N_symbs_per_subframe*N_sc_rb*M_pusch_rb);
	
	cufftComplex* ifft_vec_d;
	cufftComplex *pusch_bb_d = 0;
	cudaMalloc((void **)&ifft_vec_d, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&pusch_bb_d, sizeof(cufftComplex)*modulated_subframe_length);

	//stopTimer("Allocation Time= %.6f ms\n", elapsed);


	//timer_test << <1, 1 >> > ();

	//startTimer();

	//Generate Pseudo Random Seq.
	Byte *c_h = 0;
	generate_psuedo_random_seq(&c_h, N_bits, n_RNTI, n_s, N_id_cell);

	//Copy (c) to Device
	cudaMemcpy(c_d, c_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);

	//Interleaver
	interleaver(inputBits_d, riBits_d, &interleaved_d, N_bits, N_ri, Qm, N_l, y_idx_d, y_mat_d);

	//Scrambler
	scrambler(interleaved_d, &scrambledbits_d, c_d, N_bits+N_ri);

	//Mapper
	mapper(scrambledbits_d, N_bits+N_ri, Qm, &symbols_R_d, &symbols_I_d, bits_each_Qm_d);

	//Transform Precoder
	transform_precoder(symbols_R_d, symbols_I_d, M_pusch_rb, (N_bits+N_ri)/Qm, &precoded_symbols_d, plan_transform_precoder, cuComplex_symbols_d);

	//Generate DMRS
	generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 0, &dmrs_1_d, &dmrs_2_d, x_q_d);

	//Multiplexing the DMRS with the Data
	compose_subframe(precoded_symbols_d, dmrs_1_d, dmrs_2_d, M_pusch_rb, &subframe_d);

	// Generate SC-FDMA signal
	sc_fdma_modulator(subframe_d, M_pusch_rb, &pusch_bb_d, plan_sc_fdma, ifft_vec_d);


	//timer_test << <1, 1 >> > ();
	startTimer();
	cufftComplex *pusch_bb_h = (cufftComplex *)malloc(sizeof(cufftComplex)*(30720));
	cudaMemcpy(pusch_bb_h, pusch_bb_d, sizeof(cufftComplex)*(30720), cudaMemcpyDeviceToHost);
	stopTimer("Time= %.6f ms\n", elapsed);

	////To compare with MATLAB results
	////Run the file (output.m)
	//int NNN = modulated_subframe_length;
	//FILE *results;
	//if ((results = freopen("output.m", "w+", stdout)) == NULL) {
	//	printf("Cannot open file.\n");
	//	exit(1);
	//}

	//printf("clear; clc;\nsymbols_real = [ ");
	//for (int i = 0; i < NNN; i++)
	//{
	//	printf("%10f", pusch_bb_h[i].x);
	//	if (i != (NNN -1))
	//		printf(",");
	//}

	//printf(" ];\nsymbols_imag = [ ");

	//for (int i = 0; i < NNN; i++)
	//{
	//	printf("%10f", pusch_bb_h[i].y);
	//	if (i != (NNN -1))
	//		printf(",");
	//}

	//printf(" ];\n");
	//printf("symbols_CUDA = symbols_real + 1i * symbols_imag;\n");

	////Matlab code
	//printf("matlab_test");
	////printf("N_l = %d; \nQ_m = %d; \ndata_bits = (fread(fopen('%s')) - '0').';\nri_bits = (fread(fopen('%s'))-'0').'; \n\ninterleaved_bits = channel_interleaver(data_bits, ri_bits, [], Q_m, N_l); \nc_init = 10 * 2 ^ 14 + floor(0 / 2) * 2 ^ 9 + 2; \nc = generate_psuedo_random_seq(c_init, %d); \nb_scrampled = scrambler(interleaved_bits, c); \nsymbols_MATLAB = mapper(b_scrampled, '64qam'); \nsymbols_MATLAB = transform_precoder(symbols_MATLAB, %d); \nsum(abs(round(symbols_MATLAB, 6) - round(symbols_CUDA, 6)))",N_l,Qm, argv[1], argv[2], N_bits+N_ri,M_pusch_rb);
	//
	//fclose(results);

	//Trans Prec
	//FILE *results;
	//if ((results = freopen("output.m", "w+", stdout)) == NULL) {
	//	printf("Cannot open file.\n");
	//	exit(1);
	//}

	//printf("clear; clc;\nsymbols_real = [ ");
	//for (int i = 0; i < ((N_bits+N_ri) / Qm); i++)
	//{
	//	printf("%10f", precoded_symbols_h[i].x);
	//	if (i != (((N_bits + N_ri) / Qm) - 1))
	//		printf(",");
	//}

	//printf(" ];\nsymbols_imag = [ ");

	//for (int i = 0; i < ((N_bits + N_ri) / Qm); i++)
	//{
	//	printf("%10f", precoded_symbols_h[i].y);
	//	if (i != (((N_bits + N_ri) / Qm) - 1))
	//		printf(",");
	//}

	//printf(" ];\n");
	//printf("symbols_CUDA = symbols_real + 1i * symbols_imag;\n");

	////Matlab code
	//printf("N_l = %d; \nQ_m = %d; \ndata_bits = (fread(fopen('D:\\input_86400.txt')) - '0').';\nri_bits = (fread(fopen('D:\\ri_72.txt'))-'0').'; \n\ninterleaved_bits = channel_interleaver(data_bits, ri_bits, [], Q_m, N_l); \nc_init = 10 * 2 ^ 14 + floor(0 / 2) * 2 ^ 9 + 2; \nc = generate_psuedo_random_seq(c_init, %d); \nb_scrampled = scrambler(interleaved_bits, c); \nsymbols_MATLAB = mapper(b_scrampled, '64qam'); \nsymbols_MATLAB = transform_precoder(symbols_MATLAB, %d); \nsum(abs(round(symbols_MATLAB, 6) - round(symbols_CUDA, 6)))", N_l, Qm, N_bits + N_ri, M_pusch_rb);

	//fclose(results);


	//Interleaver
	//FILE *results;
	//if ((results = freopen("interleaver_Results.m", "w+", stdout)) == NULL) {
	//	printf("Cannot open file.\n");
	//	exit(1);
	//}

	//printf("clear; clc;\noutput = [ ");
	//for (int i = 0; i < (N_bits + N_ri); i++)
	//{
	//	printf("%d", interleaved_h[i]);
	//	if (i != (N_bits + N_ri - 1))
	//		printf(",");
	//}

	//printf(" ];\n");

	////Matlab code
	//printf("N_l = 1;\nQ_m = 6;\ndata_bits = (fread(fopen('%s'))-'0').';\nri_bits = (fread(fopen('%s'))-'0').';\noutput_MATLAB = channel_interleaver(data_bits, ri_bits, [], N_l, Q_m);\nsum(abs(output_MATLAB-output))", argv[1], argv[2]);
	//fclose(results);

}