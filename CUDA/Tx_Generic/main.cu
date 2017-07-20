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

//NOTE: RI control bits number should be set to 0 
//as we didn't care for them when we created the generic Tx as we had little time.

//Just set N_l to the required value:
// 1 ---> SISO
// 2 ---> 2x2 MIMO
// 4 ---> 4x4 MIMO
// 8 ---> 8x8 MIMO (wasn't tested but it should work)
// 64 ---> 64x64 MIMO

//You should pass these parameters through command prompt:
// input_file ---> input_1x1.txt or input_2x2.txt or input_4x4.txt or input_64x64.txt
// ri_bits file ---> ri_0.txt

int main(int argc, char **argv) {

	//For timing purpose
	timerInit();

	const int Qm = 6;					// 64QAM Modulation
	const int N_l = 4;					// Number of Layers

	int N_bits = 86400 * N_l, N_ri = 0;

	// Physical layer cell identity (we need for generation of random sequence)
	int N_id_cell = 2;						// assume enodeB scheduled cell 2 for the UE
	int M_pusch_rb = 100;					// number of resource blocks assigned to the UE
	int n_s = 0;							// assume UE send on time slot 4
	int n_RNTI = 10;						// radio network temporary identifier given to the UE by enodeB(assume 10)
	int N_subfr = 0;						// Subframe number within a radio frame
	
	//cudaMalloc & cudaMemcpy for inputBits & RI_Bits to Device
	Byte *inputBits_d = 0, *riBits_d = 0;

	cudaMalloc((void **)&inputBits_d, sizeof(Byte)*N_bits);
	cudaMalloc((void **)&riBits_d, sizeof(Byte)*N_ri);
	Byte* c_d = 0;
	cudaMalloc((void **)&c_d, sizeof(Byte)*N_bits);

	startTimer();
	BYTE* inputBits_h = readBits(argc, argv[1], &N_bits);				//Get input bits from the text file
	BYTE* riBits_h = readBits(argc, argv[2], &N_ri);					//Get RI bits from the text file

	cudaMemcpyAsync(inputBits_d, inputBits_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(riBits_d, riBits_h, sizeof(Byte)*N_ri, cudaMemcpyHostToDevice);
	stopTimer("cudaMalloc & cudaMemcpy for inputBits & RI_Bits Time= %.6f ms\n");

	//Create Plans
	startTimer();

	cufftHandle plan_transform_precoder[N_l];
	cufftHandle plan_sc_fdma[N_l];

	int n[1] = { N_sc_rb*M_pusch_rb };
	int n_2[1] = { FFT_size };

	for (int i = 0; i < N_l; i++)
	{
		cufftPlanMany(&plan_transform_precoder[i], 1, n, NULL, 1, n[0], NULL, 1, N_sc_rb*M_pusch_rb, CUFFT_C2C, (((N_bits / N_l) + N_ri) / Qm) / n[0]);
		cufftPlanMany(&plan_sc_fdma[i], 1, n_2, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);
	}

	stopTimer("Create Plans Time= %.6f ms\n");

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

	//Note: If you create streams, y_idx_d & y_mat_d should be duplicated

	Byte *interleaved_d_total;

	Byte *scrambledbits_d[N_l];

	Byte *bits_each_Qm_d[N_l];
	float* symbols_R_d[N_l];
	float* symbols_I_d[N_l];

	cufftComplex *precoded_symbols_d[N_l];
	cufftComplex *cuComplex_symbols_d[N_l];

	cufftComplex* x_q_d[N_l];
	cufftComplex* dmrs_d_1[N_l];
	cufftComplex *dmrs_d_2[N_l];
	cufftComplex *subframe_d[N_l];
	
	cufftComplex* ifft_vec_d[N_l];
	cufftComplex *pusch_bb_d[N_l];

	cudaMalloc((void **)&y_idx_d, sizeof(Byte)*(N_pusch_symbs * R_prime_mux));
	cudaMalloc((void **)&y_mat_d, sizeof(Byte)*(N_pusch_symbs*R_mux));
	cudaMalloc((void **)&interleaved_d_total, sizeof(Byte)*(N_pusch_symbs*R_mux));

	for (int i = 0; i < N_l; i++)
	{
		cudaMalloc((void **)&scrambledbits_d[i], sizeof(Byte)*N_bits / N_l);
		cudaMalloc((void **)&bits_each_Qm_d[i], sizeof(Byte)*(N_bits / (Qm*N_l)));
		cudaMalloc((void **)&symbols_R_d[i], sizeof(float)*(N_bits / (Qm*N_l)));
		cudaMalloc((void **)&symbols_I_d[i], sizeof(float)*(N_bits / (Qm*N_l)));
		cudaMalloc((void **)&cuComplex_symbols_d[i], sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
		cudaMalloc((void **)&precoded_symbols_d[i], sizeof(cufftComplex)*(N_bits / (Qm*N_l)));
		cudaMalloc((void **)&dmrs_d_1[i], sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
		cudaMalloc((void **)&dmrs_d_2[i], sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
		cudaMalloc((void **)&x_q_d[i], sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);
		cudaMalloc((void **)&subframe_d[i], sizeof(cufftComplex)*N_symbs_per_subframe*N_sc_rb*M_pusch_rb);
		cudaMalloc((void **)&ifft_vec_d[i], sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
		cudaMalloc((void **)&pusch_bb_d[i], sizeof(cufftComplex)*modulated_subframe_length);

	}

	stopTimer("Device data allocation Time= %.6f ms\n");

	cufftComplex *pusch_bb_h[N_l];

	for (int i = 0; i < N_l; i++)
	{
		pusch_bb_h[i] = (cufftComplex *)malloc(sizeof(cufftComplex)*(modulated_subframe_length));
	}

	startTimer();

	//Generate Pseudo Random Seq.
	Byte *c_h = 0;
	generate_psuedo_random_seq(&c_h, N_bits, n_RNTI, n_s, N_id_cell);
	
	//Copy (c) to Device
	cudaMemcpyAsync(c_d, c_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);
	
	//Interleaver
	//Interleaver will be modified from inside in higher order of MIMO
	//This is interleaver on CPU code, RI is not done in this version
	//startTimer();
	interleaver(inputBits_d, riBits_d, &interleaved_d_total, N_bits, N_ri, Qm, N_l, y_idx_d, y_mat_d);

	//int NZ = 100;
	//Byte* hprint = (Byte *)malloc(sizeof(Byte)*(NZ));
	//cudaMemcpy(hprint, interleaved_d_total, sizeof(Byte)*(NZ), cudaMemcpyDeviceToHost);
	//stopTimer("Interleaver Time= %.6f ms\n");


	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%d", hprint[i]);
	//}

	//Scrambler
	for (int i = 0; i < N_l; i++)
	{
		scrambler(interleaved_d_total + (i * N_bits / N_l), &scrambledbits_d[i], c_d, (N_bits / N_l) + N_ri);
	}

	//int NZ = 100;
	//Byte* hprint = (Byte *)malloc(sizeof(Byte)*(NZ));
	//cudaMemcpy(hprint, scrambledbits_d[0], sizeof(Byte)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%d", hprint[i]);
	//}

	//Mapper
	for (int i = 0; i < N_l; i++)
	{
		mapper(scrambledbits_d[i], (N_bits / N_l) + N_ri, Qm, &symbols_R_d[i], &symbols_I_d[i], bits_each_Qm_d[i]);
	}

	//int NZ = 100;
	//float* hprint = (float *)malloc(sizeof(float)*(NZ));
	//cudaMemcpy(hprint, symbols_R_d[2], sizeof(float)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%10f", hprint[i]);
	//}

	//Transform Precoder
	for (int i = 0; i < N_l; i++)
	{
		transform_precoder(symbols_R_d[i], symbols_I_d[i], M_pusch_rb, ((N_bits / N_l) + N_ri) / Qm, &precoded_symbols_d[i], plan_transform_precoder[i], cuComplex_symbols_d[i]);
	}

	//int NZ = 100;
	//cufftComplex* hprint = (cufftComplex *)malloc(sizeof(cufftComplex)*(NZ));
	//cudaMemcpy(hprint, precoded_symbols_d[3], sizeof(cufftComplex)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%10f", hprint[i].x);
	//}

	//Generate DMRS
	for (int i = 0; i < N_l; i++)
	{	
		generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, (i%4), &dmrs_d_1[i], &dmrs_d_2[i], x_q_d[i]);
	}
	
	//int NZ = 100;
	//cufftComplex* hprint = (cufftComplex *)malloc(sizeof(cufftComplex)*(NZ));
	//cudaMemcpy(hprint, dmrs_d_1[3], sizeof(cufftComplex)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%10f", hprint[i].x);
	//}

	//Multiplexing the DMRS with the Data
	for (int i = 0; i < N_l; i++)
	{	
		compose_subframe(precoded_symbols_d[i], dmrs_d_1[i], dmrs_d_2[i], M_pusch_rb, &subframe_d[i], N_l);
	}

	//int NZ = 100;
	//cufftComplex* hprint = (cufftComplex *)malloc(sizeof(cufftComplex)*(NZ));
	//cudaMemcpy(hprint, subframe_d[0], sizeof(cufftComplex)*(NZ), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < NZ; i++)
	//{
	//	printf("%10f", hprint[i].x);
	//}

	// Generate SC-FDMA signal

	for (int i = 0; i < N_l; i++)
	{	
		sc_fdma_modulator(subframe_d[i], M_pusch_rb, &pusch_bb_d[i], plan_sc_fdma[i], ifft_vec_d[i]);
		cudaMemcpy(pusch_bb_h[i], pusch_bb_d[i], sizeof(cufftComplex)*(modulated_subframe_length), cudaMemcpyDeviceToHost);
	}

	stopTimer("Processing Time= %.6f ms\n");

#pragma region Results Printing

	//To compare with MATLAB results
	//Run the file (output.m)
	int NNN = modulated_subframe_length;
	FILE *results;
	if ((results = freopen("output.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	printf("clear; clc;");

	for (int j = 0; j < N_l; j++)
	{
		printf("\nsymbols_real = [ ");
		for (int i = 0; i < NNN; i++)
		{
			printf("%10f", pusch_bb_h[j][i].x);
			if (i != (NNN - 1))
				printf(",");
		}

		printf(" ];\nsymbols_imag = [ ");

		for (int i = 0; i < NNN; i++)
		{
			printf("%10f", pusch_bb_h[j][i].y);
			if (i != (NNN - 1))
				printf(",");
		}

		printf(" ];\n");
		printf("symbols_CUDA_%d = symbols_real + 1i * symbols_imag;\n", j + 1);

	}

	printf("symbols_CUDA = [");
	for (int j = 0; j < N_l; j++)
	{
		printf("symbols_CUDA_%d ", j + 1);
	}

	printf("];\n");

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
	printf("N_l = %d; \nQ_m = %d; \ndata_bits_total = (fread(fopen('%s')) - '0').';\ndata_bits = reshape(data_bits_total,length(data_bits_total)/N_l,N_l);\nri_bits = (fread(fopen('%s'))-'0').';\n", N_l, Qm, argv[1], argv[argc - 1]);
	printf("interleaved_bits = channel_interleaver_MIMO(data_bits, ri_bits, [], N_l, Q_m);\ninterleaved_bits_Nlayer_col = reshape(interleaved_bits,length(interleaved_bits)/N_l,N_l);\nc_init = 10 * 2 ^ 14 + floor(0 / 2) * 2 ^ 9 + 2; \nc = generate_psuedo_random_seq(c_init, length(interleaved_bits_Nlayer_col)); \nscrambled = scrambler_MIMO(interleaved_bits_Nlayer_col.', repmat(c,N_l,1), N_l);for i = 1:N_l\n	modulated_symbols(:,i) = mapper(scrambled(i,:), mod_type).';\nend\n");
	if (N_l != 1)		//MIMO
	{
		printf("\ntransform_precoded_symbols = transform_precoder_mimo(modulated_symbols, M_pusch_rb, N_l);\nprecoded_symbols = precoding_mimo(transform_precoded_symbols, N_l, N_l); \nfor i = 1:N_l\n	dmrs(i, :) = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, mod((i - 1), 4)); \nend\ndmrs1 = dmrs(:, 1 : M_pusch_sc); \ndmrs2 = dmrs(:, M_pusch_sc + 1 : 2 * M_pusch_sc); \nsubframe_per_ant = compose_subframe_mimo(precoded_symbols, dmrs1, dmrs2, M_pusch_rb, N_l); \nsymbols_MATLAB = sc_fdma_modulator_MIMO(subframe_per_ant, M_pusch_rb, N_l); \nsymbols_MATLAB_reshaped = reshape(symbols_MATLAB.',1,length(symbols_MATLAB)*N_l);");
	}
	else				//SISO
	{
		printf("precoded_data = transform_precoder(modulated_symbols, M_pusch_rb);\ndmrs = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 0);\ndmrs_1 = dmrs(1:M_pusch_sc);\ndmrs_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);\nsubframe_1 = compose_subframe(precoded_data, dmrs_1, dmrs_2, M_pusch_rb);\nsymbols_MATLAB = sc_fdma_modulator(subframe_1, M_pusch_rb);\nsymbols_MATLAB_reshaped = reshape(symbols_MATLAB.',1,length(symbols_MATLAB)*N_l);");
	}
	printf("\n\nsum((abs(symbols_MATLAB_reshaped - symbols_CUDA)))");

	fclose(results);

#pragma endregion

	}
