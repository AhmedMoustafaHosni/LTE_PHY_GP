
#include "main.cuh"


int main(int argc, char **argv) {

	// Create default stream for the chain
	cudaStream_t stream_dmrs;
	cudaStreamCreate(&stream_default);
	cudaStreamCreate(&stream_dmrs);

	//For timing purpose
	timerInit();
	startTimer();

	int N_bits, N_ri;
	const int Qm = 6;					// 64QAM Modulation
	const int N_l = 1;					// Number of Layers

	// Physical layer cell identity (we need for generation of random sequence)
	int N_id_cell = 2;						// assume enodeB scheduled cell 2 for the UE
	int M_pusch_rb = 100;					// number of resource blocks assigned to the UE
	int n_s = 0;							// assume UE send on time slot 0
	int n_RNTI = 10;						// radio network temporary identifier given to the UE by enodeB(assume 10)
	// (UNUSED) int N_subfr = 0;						// Subframe number within a radio frame
	BYTE* inputBits_h = readBits(argc, argv[1], &N_bits);			//Get input bits from the text file
	BYTE* riBits_h = readBits(argc, argv[2], &N_ri);					//Get RI bits from the text file

	//cudaMalloc & cudaMemcpy for inputBits & RI_Bits to Device
	Byte *inputBits_d = 0, *riBits_d = 0;

	cudaMalloc((void **)&inputBits_d, sizeof(Byte)*N_bits);
	cudaMalloc((void **)&riBits_d, sizeof(Byte)*N_ri);
	Byte* c_d = 0;
	cudaMalloc((void **)&c_d, sizeof(Byte)*N_bits);
	
	// Copy data to the device using different stream 
	cudaStream_t stream_mem;
	cudaStreamCreate(&stream_mem);
	cudaMemcpyAsync(inputBits_d, inputBits_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice, stream_mem);
	cudaMemcpyAsync(riBits_d, riBits_h, sizeof(Byte)*N_ri, cudaMemcpyHostToDevice, stream_default);
	stopTimer("cudaMalloc & cudaMemcpy for inputBits & RI_Bits Time= %.6f ms\n", elapsed);

	//Create Plans
	startTimer();
	cufftHandle plan_transform_precoder;
	int n[1] = { N_sc_rb*M_pusch_rb };
	cufftPlanMany(&plan_transform_precoder, 1, n, NULL, 1, n[0], NULL, 1, N_sc_rb*M_pusch_rb, CUFFT_C2C, ((N_bits + N_ri) / Qm)/n[0]);

	cufftHandle plan_sc_fdma;
	n[0] = { FFT_size };
	cufftPlanMany(&plan_sc_fdma, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);
	stopTimer("Create Plans Time= %.6f ms\n", elapsed);

	//Device data allocation
	startTimer();

	//timer_test << <1, 1 >> > ();

	int data_vec_len = Qm*N_l;
	// (UNUSED) int ri_vec_len = Qm*N_l;
	int N_data_bits = N_bits / data_vec_len;
	int N_ri_bits = N_ri / data_vec_len;
	int H_prime = N_data_bits;
	// (UNUSED) int H_vec_len = data_vec_len;
	int H_prime_total = H_prime + N_ri_bits;

	int R_mux = (H_prime_total*Qm*N_l) / N_pusch_symbs;
	int R_prime_mux = R_mux / (Qm*N_l);

	Byte *y_idx_d, *y_mat_d, *interleaved_d;
	cudaMalloc((void **)&y_idx_d, sizeof(Byte)*(N_pusch_symbs * R_prime_mux));
	cudaMalloc((void **)&y_mat_d, sizeof(Byte)*(N_pusch_symbs*R_mux));
	cudaMalloc((void **)&interleaved_d, sizeof(Byte)*(N_pusch_symbs*R_mux));

	Byte *scrambledbits_d = 0;
	cudaMalloc((void **)&scrambledbits_d, sizeof(Byte)*N_bits);

	Byte *bits_each_Qm_d;
	//float* symbols_R_d = 0, *symbols_I_d = 0;
	cudaMalloc((void **)&bits_each_Qm_d, sizeof(Byte)*(N_bits / Qm));
	//cudaMalloc((void **)&symbols_R_d, sizeof(float)*(N_bits / Qm));
	//cudaMalloc((void **)&symbols_I_d, sizeof(float)*(N_bits / Qm));

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
	stopTimer("Device data allocation Time= %.6f ms\n", elapsed);

	/*startTimer();
	stopTimer("Overhead of timer= %.6f ms\n", elapsed);*/
	int times = 1;
	startTimer();
	//Generate Pseudo Random Seq.
	Byte *c_h = 0;
	generate_psuedo_random_seq(&c_h, N_bits, n_RNTI, n_s, N_id_cell);

	//Copy (c) to Device
	cudaMemcpyAsync(c_d, c_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice, stream_default);

	//for (int i = 0; i < times; i++)
	//{
		//Interleaver
		interleaver(inputBits_d, riBits_d, &interleaved_d, N_bits, N_ri, Qm, N_l, y_idx_d, y_mat_d);

		//Generate DMRS
		generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 0, &dmrs_1_d, &dmrs_2_d, x_q_d, stream_dmrs);

		//Scrambler
		scrambler(interleaved_d, &scrambledbits_d, c_d, N_bits + N_ri);

		//Mapper
		mapper(scrambledbits_d, N_bits + N_ri, Qm, M_pusch_rb, cuComplex_symbols_d, bits_each_Qm_d); // Mohammed

		//Transform Precoder
		transform_precoder(&precoded_symbols_d, plan_transform_precoder, cuComplex_symbols_d);
		//Multiplexing the DMRS with the Data
		compose_subframe(precoded_symbols_d, dmrs_1_d, dmrs_2_d, M_pusch_rb, &subframe_d);

		// Generate SC-FDMA signal
		sc_fdma_modulator(subframe_d, M_pusch_rb, &pusch_bb_d, plan_sc_fdma, ifft_vec_d);

		//timer_test << <1, 1 >> > ();
		//startTimer();
		cufftComplex *pusch_bb_h = (cufftComplex *)malloc(sizeof(cufftComplex)*(30720));
		cudaMemcpyAsync(pusch_bb_h, pusch_bb_d, sizeof(cufftComplex)*(30720), cudaMemcpyDeviceToHost, stream_default);
	//}
	stopTimer("Processing Time= %.6f ms\n", elapsed/ times);

	//To compare with MATLAB results
	//Run the file (output.m)
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

	//fclose(results);

	//if ((results = freopen("matlab_test.m", "w+", stdout)) == NULL) {
	//	printf("Cannot open file.\n");
	//	exit(1);
	//}

	//printf("N_bits = %d; \n", N_bits);
	//if(Qm == 6)
	//	printf("mod_type = %s; \n", "'64qam'");
	//else if (Qm == 4)
	//	printf("mod_type = %s; \n", "'16qam'");
	//else if (Qm == 2)
	//	printf("mod_type = %s; \n", "'qpsk'");
	//else if (Qm == 1)
	//	printf("mod_type = %s; \n", "'bpsk'");
	//
	//printf("N_sc_rb   = 12;      %% number of subcarriers in each resource block\n");
	//printf("M_pusch_rb = %d;      %% number of resource blocks assigned to the UE\n", M_pusch_rb);
	//printf("M_pusch_sc = M_pusch_rb*N_sc_rb;  %% total number of subcarriers\n\n");
	//printf("N_l = %d; \nQ_m = %d; \ndata_bits = (fread(fopen('%s')) - '0').';\nri_bits = (fread(fopen('%s'))-'0').'; \n", N_l, Qm, argv[1], argv[2]);
	//printf("interleaved_bits = channel_interleaver(data_bits, ri_bits, [], Q_m, N_l); \nc_init = 10 * 2 ^ 14 + floor(0 / 2) * 2 ^ 9 + 2; \nc = generate_psuedo_random_seq(c_init, N_bits); \nb_scrampled = scrambler(interleaved_bits, c); \nmapped = mapper(b_scrampled, mod_type); \nprecoded_data = transform_precoder(mapped, M_pusch_rb); \n\ndmrs = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 0);\ndmrs_1 = dmrs(1:M_pusch_sc);\ndmrs_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);\nsubframe_1 = compose_subframe(precoded_data, dmrs_1, dmrs_2, M_pusch_rb);\nsymbols_MATLAB = sc_fdma_modulator(subframe_1, M_pusch_rb);\n\nsum((abs(symbols_MATLAB) - abs(symbols_CUDA)))");

	//fclose(results);

	// Free allocated memory
	// free device arrays
	cudaFree(inputBits_d);
	cudaFree(riBits_d);
	cudaFree(c_d);
	cudaFree(y_idx_d);
	cudaFree(y_mat_d);
	cudaFree(interleaved_d);
	cudaFree(scrambledbits_d);
	cudaFree(bits_each_Qm_d);
	cudaFree(cuComplex_symbols_d);
	cudaFree(precoded_symbols_d);
	cudaFree(dmrs_1_d);
	cudaFree(dmrs_2_d);
	cudaFree(x_q_d);
	cudaFree(subframe_d);
	cudaFree(ifft_vec_d);
	cudaFree(pusch_bb_d);

	// free host arrays 
	//free(inputBits_h);

}