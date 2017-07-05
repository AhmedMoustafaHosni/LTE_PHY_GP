/*
% Function:    receiver
By: Mohammed Osama & Khaled Ahmed 
*/

#include "sc_fdma_demodulator.cuh"
#include "generate_dmrs_pusch.cuh"
#include "generate_ul_rs.cuh"
#include "generate_psuedo_random_seq.cuh"
#include "transform_predecoder.cuh"
#include "decompose_subframe.cuh"
#include "demapper.cuh"
#include "descrambler.cuh"
#include "deinterleaver.cuh"
#include "channel_estimation.cuh"
#include "channel_equalization_zf.cuh"


int main(int argc, char **argv) {

	//input
	cufftComplex* subframe_h = (cufftComplex *)malloc(sizeof(cufftComplex)*modulated_subframe_length);
	cufftComplex* subframe_h2 = (cufftComplex *)malloc(sizeof(cufftComplex)*modulated_subframe_length);

	for (int i = 0; i < modulated_subframe_length; i++)
	{
		subframe_h[i].x = rand() / (float)RAND_MAX * 0.5;
		subframe_h[i].y = rand() / (float)RAND_MAX * 0.5;
	}

	for (int i = 0; i < modulated_subframe_length; i++)
	{
		subframe_h2[i].x = rand() / (float)RAND_MAX * 0.5;
		subframe_h2[i].y = rand() / (float)RAND_MAX * 0.5;
	}


	//For timing purpose
	timerInit();
	startTimer();

	const int Qm = 6;				// Modulation Order(2 = QPSK, 4 = 16QAM, 6 = 64QAM)
	const int M_pusch_rb = 100;		//number of resource blocks assigned to the UE
	const int N_l = 2;				// Number of Layers
	const int N_ri = 0;			//length of ri symbols
	const int n_s = 0;				//assume UE send on subframe 0
	const int N_id_cell = 2;		//assume enodeB scheduled cell 2 for the UE
	const int M_pusch_sc = N_sc_rb * M_pusch_rb; //total number of subcarriers
	const int n_RNTI = 10;			//radio network temporary identifier given to the UE by enodeB (assume 10)
	const int N_bits = Qm * 12 * M_pusch_sc;   //Qm * 12 * M_pusch_sc = 2*12*1200

	cufftComplex* subframe_d;
	cufftComplex* subframe_d2;
	cudaMalloc((void **)&subframe_d, sizeof(cufftComplex)*modulated_subframe_length);
	cudaMalloc((void **)&subframe_d2, sizeof(cufftComplex)*modulated_subframe_length);
	cudaMemcpy(subframe_d, subframe_h, sizeof(cufftComplex)*modulated_subframe_length, cudaMemcpyHostToDevice);
	cudaMemcpy(subframe_d2, subframe_h2, sizeof(cufftComplex)*modulated_subframe_length, cudaMemcpyHostToDevice);

	//Generate Pseudo Random Seq.
	Byte *c_h = 0;
	generate_psuedo_random_seq(&c_h, N_bits, n_RNTI, n_s, N_id_cell);
	//Copy (c) to Device
	Byte* c_d = 0;
	cudaMalloc((void **)&c_d, sizeof(Byte)*Qm * 12 * M_pusch_sc);
	cudaMemcpyAsync(c_d, c_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);
	stopTimer("Time of copying of data to device= %.6f ms\n", elapsed);


	startTimer();
	//Device data allocation
	cufftComplex* fft_vec_d;
	cufftComplex* fft_vec_d2;
	cufftComplex* demod_subframe_d;
	cufftComplex* demod_subframe_d2;
	cufftComplex* demod_subframe_h = (cufftComplex*)malloc(sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc);
	cufftComplex* demod_subframe_h2 = (cufftComplex*)malloc(sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc);

	cudaMalloc((void **)&fft_vec_d, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&fft_vec_d2, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&demod_subframe_d, sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc);
	cudaMalloc((void **)&demod_subframe_d2, sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc);

	cufftComplex* x_q_d;
	cufftComplex* x_q_d2;
	cufftComplex* dmrs1_generated_d = 0, *dmrs2_generated_d = 0;
	cufftComplex* dmrs1_generated_d2 = 0, *dmrs2_generated_d2 = 0;
	cudaMalloc((void **)&dmrs1_generated_d, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs2_generated_d, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs1_generated_d2, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs2_generated_d2, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);

	cudaMalloc((void **)&x_q_d, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);
	cudaMalloc((void **)&x_q_d2, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);

	cufftComplex* dmrs1_decomposed_d;
	cufftComplex* dmrs2_decomposed_d;
	cufftComplex* dmrs1_decomposed_d2;
	cufftComplex* dmrs2_decomposed_d2;
	cufftComplex* complex_data_d;
	cufftComplex* complex_data_d2;
	cufftComplex* complex_data_h = (cufftComplex*)malloc(sizeof(cufftComplex)* 12 * M_pusch_sc);
	cufftComplex* complex_data_h2 = (cufftComplex*)malloc(sizeof(cufftComplex)* 12 * M_pusch_sc);
	cudaMalloc((void **)&complex_data_d, sizeof(cufftComplex)* 12 * M_pusch_sc);
	cudaMalloc((void **)&complex_data_d2, sizeof(cufftComplex)* 12 * M_pusch_sc);
	cudaMalloc((void **)&dmrs1_decomposed_d, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&dmrs1_decomposed_d2, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&dmrs2_decomposed_d, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&dmrs2_decomposed_d2, sizeof(cufftComplex)*M_pusch_sc);


	// Channel estimation and equaliuzation allocation
	/*cufftComplex* channel, *equalized_subframe_d;
	cudaMalloc((void **)&channel, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&equalized_subframe_d, sizeof(cufftComplex)* 12 * M_pusch_sc);

	cufftComplex* equalized_subframe_h = (cufftComplex*)malloc(sizeof(cufftComplex)* 12 * M_pusch_sc);

	cufftComplex* channel_h = (cufftComplex*)malloc(sizeof(cufftComplex)*M_pusch_sc);*/


	cufftComplex* predecoded_data_d;
	cudaMalloc((void **)&predecoded_data_d, sizeof(cufftComplex)* 12 * M_pusch_sc);
	cufftComplex* predecoded_data_d2;
	cudaMalloc((void **)&predecoded_data_d2, sizeof(cufftComplex)* 12 * M_pusch_sc);

	Byte *bits_d;
	cudaMalloc((void **)&bits_d, sizeof(Byte)* Qm * 12 * M_pusch_sc);    //FIX Number_demaped_bits
	Byte *bits_d2;
	cudaMalloc((void **)&bits_d2, sizeof(Byte)* Qm * 12 * M_pusch_sc);    //FIX Number_demaped_bits

	Byte *descrambled_bits_d;
	cudaMalloc((void **)&descrambled_bits_d, sizeof(Byte)* Qm * 12 * M_pusch_sc);
	Byte *descrambled_bits_d2;
	cudaMalloc((void **)&descrambled_bits_d2, sizeof(Byte)* Qm * 12 * M_pusch_sc);
	Byte *descrambled_bits_h = (Byte *)malloc(sizeof(Byte)* Qm * 12 * M_pusch_sc);
	Byte *descrambled_bits_h2 = (Byte *)malloc(sizeof(Byte)* Qm * 12 * M_pusch_sc);


	// Step 1: Define C_mux
	int C_mux = N_pusch_symbs;
	// Step 2: Define R_mux and R_prime_mux
	int H_prime_total = N_bits * N_l / (Qm*N_l);
	int H_prime = H_prime_total - N_ri;
	int R_mux = (H_prime_total*Qm*N_l) / C_mux;
	int R_prime_mux = R_mux / (Qm*N_l);

	Byte *ri_d, *y_idx_d, *y_mat_d;
	Byte *received_bits_d;
	//Byte *received_bits_h = (Byte *)malloc(sizeof(Byte *) * N_bits);
	cudaMalloc((void **)&ri_d, sizeof(Byte)*(N_ri * Qm * N_l));
	cudaMalloc((void **)&y_idx_d, sizeof(Byte)*(C_mux*R_prime_mux));
	cudaMalloc((void **)&y_mat_d, sizeof(Byte)*(C_mux*R_mux));
	cudaMalloc((void **)&received_bits_d, sizeof(Byte)* H_prime * Qm * N_l);
	stopTimer("Allocation Time= %.6f ms\n", elapsed);

	startTimer();
	//create plans
	int n[1] = { FFT_size };
	cufftHandle plan_sc_fdma, plan_sc_fdma2;
	cufftPlanMany(&plan_sc_fdma, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);
	cufftPlanMany(&plan_sc_fdma2, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);

	int N_SIGS = 12;   //signal_size/M_pusch_sc = 12 * M_pusch_sc / M_pusch_sc = 12
	n[0] = { M_pusch_sc };
	cufftHandle plan_transform_predecoder;
	cufftHandle plan_transform_predecoder2;
	cufftPlanMany(&plan_transform_predecoder, 1, n, NULL, 1, M_pusch_sc, NULL, 1, M_pusch_sc, CUFFT_C2C, N_SIGS);
	cufftPlanMany(&plan_transform_predecoder2, 1, n, NULL, 1, M_pusch_sc, NULL, 1, M_pusch_sc, CUFFT_C2C, N_SIGS);
	stopTimer("Time of plan creation= %.6f ms\n", elapsed);

	startTimer();

	//sc-fdma demodulation
	sc_fdma_demodulator(subframe_d, M_pusch_rb, &demod_subframe_d, plan_sc_fdma, fft_vec_d);
	sc_fdma_demodulator(subframe_d2, M_pusch_rb, &demod_subframe_d2, plan_sc_fdma2, fft_vec_d2);

	//cudaMemcpy(demod_subframe_h, demod_subframe_d, sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc, cudaMemcpyDeviceToHost);
	//cudaMemcpy(demod_subframe_h2, demod_subframe_d2, sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc, cudaMemcpyDeviceToHost);
	//generate dmrs   
	//generate_dmrs_pusch(n_s, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 0, &dmrs1_generated_d, &dmrs2_generated_d, x_q_d);
	//generate_dmrs_pusch(n_s, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 1, &dmrs1_generated_d2, &dmrs2_generated_d2, x_q_d2);


	//Decompose subframe
	decompose_subframe(demod_subframe_d, M_pusch_rb, &complex_data_d, &dmrs1_decomposed_d, &dmrs2_decomposed_d);
	decompose_subframe(demod_subframe_d2, M_pusch_rb, &complex_data_d2, &dmrs1_decomposed_d2, &dmrs2_decomposed_d2);

	//Channel estimation
	//channe_estimation(dmrs1_decomposed_d, dmrs2_decomposed_d, dmrs1_generated_d, dmrs2_generated_d, M_pusch_sc, &channel);

	//cudaMemcpy(channel_h, channel, sizeof(cufftComplex)* M_pusch_sc, cudaMemcpyDeviceToHost);

	//Equalization ZF
	//channel_equalization_zf(demod_subframe_d, M_pusch_sc , channel, &equalized_subframe_d);

	//cudaMemcpy(complex_data_h, complex_data_d, sizeof(cufftComplex)* 12 * M_pusch_sc, cudaMemcpyDeviceToHost);
	//cudaMemcpy(complex_data_h2, complex_data_d2, sizeof(cufftComplex)* 12 * M_pusch_sc, cudaMemcpyDeviceToHost);

	//predecoding   
	transform_predecoder(complex_data_d, &predecoded_data_d, plan_transform_predecoder); //signal_size = 12 * M_pusch_sc
	transform_predecoder(complex_data_d2, &predecoded_data_d2, plan_transform_predecoder2); //signal_size = 12 * M_pusch_sc

	//demapping
	demapper(predecoded_data_d, M_pusch_rb, &bits_d, Qm * 12 * M_pusch_sc, Qm);  //Number_demaped_bits = Qm * 12 * M_pusch_sc
	demapper(predecoded_data_d2, M_pusch_rb, &bits_d2, Qm * 12 * M_pusch_sc, Qm);  //Number_demaped_bits = Qm * 12 * M_pusch_sc


	//Descrammpling
	descrambler(bits_d, &descrambled_bits_d, c_d, N_bits);
	descrambler(bits_d2, &descrambled_bits_d2, c_d, N_bits);
	
	//cudaMemcpy(descrambled_bits_h, descrambled_bits_d, sizeof(Byte) * N_bits, cudaMemcpyDeviceToHost);
	//cudaMemcpy(descrambled_bits_h2, descrambled_bits_d2, sizeof(Byte) * N_bits, cudaMemcpyDeviceToHost);
	
	//deinterleaver
	deinterleaver(descrambled_bits_d, descrambled_bits_d2, &ri_d, &received_bits_d, N_bits*N_l, N_ri, Qm, N_l, y_idx_d, y_mat_d);

	//cudaMemcpy(received_bits_h, received_bits_d, sizeof(Byte *) * N_bits * 2, cudaMemcpyDeviceToHost);

	//Retrieve data from device
	Byte* received_bits_h = (Byte*)malloc(sizeof(Byte)*N_bits * N_l);
	cudaMemcpy(received_bits_h, received_bits_d, sizeof(Byte)*N_bits* N_l, cudaMemcpyDeviceToHost);

	Byte* ri_h = (Byte*)malloc(sizeof(Byte)*N_ri * Qm * N_l);
	cudaMemcpy(ri_h, ri_d, sizeof(Byte)*N_ri * Qm * N_l, cudaMemcpyDeviceToHost);
	stopTimer("Time of processing= %.6f ms\n", elapsed);

	//Print results
	/*for (int i = 0; i < H_prime * Qm * N_l; i++)
	{
	printf("idx = %d \t %d  \n", i + 1, received_bits_h[i]);
	}
	*/

	//test file
	FILE *results1;
	if ((results1 = freopen("Receiver_test.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	//input subframe
	printf("clear; clc;\nsymbols_in_real = [ ");
	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", subframe_h[i].x);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}
	printf(" ];\nsymbols_in_imag = [ ");
	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", subframe_h[i].y);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}
	printf(" ];\n");
	printf("subframe_CUDA = symbols_in_real + 1i * symbols_in_imag;\n");

	printf("\nsymbols_in_real = [ ");
	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", subframe_h2[i].x);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}
	printf(" ];\nsymbols_in_imag = [ ");
	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", subframe_h2[i].y);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}
	printf(" ];\n");
	printf("subframe_input2_CUDA = symbols_in_real + 1i * symbols_in_imag;\n");
	printf("subframe_input_CUDA = [ subframe_CUDA; subframe_input2_CUDA];\n");

	//printf("subframe_input_CUDA = symbols_in_real + 1i * symbols_in_imag;\n");



	//// Channel estimation 
	//printf("x = [ ");
	//for (int i = 0; i <  M_pusch_sc; i++)
	//{
	//	printf("%f ", channel_h[i].x);
	//}
	//printf(" ]; ");
	//printf("\n");
	//printf("y = [ ");
	//for (int i = 0; i <  M_pusch_sc; i++)
	//{
	//	printf("%f ", channel_h[i].y);
	//}
	//printf(" ];\n ");
	//printf("channel_cuda = x + 1i * y;\n");
	// channel equalization 
	/*printf("x = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
	printf("%f ", equalized_subframe_h[i].x);
	}
	printf(" ]; ");
	printf("\n");
	printf("y = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
	printf("%f ", equalized_subframe_h[i].y);
	}
	printf(" ];\n ");
	printf("equalized_subframe_h = x + 1i * y;\n");*/

	// sc-fdma_demodulation 

	printf("x = [ ");
	for (int i = 0; i < (N_symbs_per_subframe*M_pusch_sc); i++)
	{
		printf("%f ", demod_subframe_h[i].x);
	}
	printf(" ]; ");
	printf("\n");
	printf("y = [ ");
	for (int i = 0; i < (N_symbs_per_subframe*M_pusch_sc); i++)
	{
		printf("%f ", demod_subframe_h[i].y);
	}
	printf(" ];\n ");
	printf("demod_subframe_h = x + 1i * y;\n");

	printf("x = [ ");
	for (int i = 0; i < (N_symbs_per_subframe*M_pusch_sc); i++)
	{
		printf("%f ", demod_subframe_h2[i].x);
	}
	printf(" ]; ");
	printf("\n");
	printf("y = [ ");
	for (int i = 0; i < (N_symbs_per_subframe*M_pusch_sc); i++)
	{
		printf("%f ", demod_subframe_h2[i].y);
	}
	printf(" ];\n ");
	printf("demod_subframe_h2 = x + 1i * y;\n");

	// test decompose subfram 
	printf("x = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		printf("%f ", complex_data_h[i].x);
	}
	printf(" ]; ");
	printf("\n");
	printf("y = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		printf("%f ", complex_data_h[i].y);
	}
	printf(" ];\n ");
	printf("complex_data_h = x + 1i * y;\n");

	printf("x = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		printf("%f ", complex_data_h2[i].x);
	}
	printf(" ]; ");
	printf("\n");
	printf("y = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		printf("%f ", complex_data_h2[i].y);
	}
	printf(" ];\n ");
	printf("complex_data_h2 = x + 1i * y;\n");

	//Received Bits
	printf("\ndescrambled_bits_cuda = [ ");
	for (int i = 0; i < (N_bits); i++)
	{
		printf("%d", descrambled_bits_h[i]);
		if (i != ((Qm * 12 * M_pusch_sc) - 1))
			printf(",");
	}
	printf(" ];\n");

	printf("\ndescrambled_bits2_cuda = [ ");
	for (int i = 0; i < (N_bits); i++)
	{
		printf("%d", descrambled_bits_h2[i]);
		if (i != ((Qm * 12 * M_pusch_sc) - 1))
			printf(",");
	}
	printf(" ];\n");


	//Received Bits
	printf("\nReceved_bits_cuda = [ ");
	for (int i = 0; i < (N_bits * N_l); i++)
	{
		printf("%d", received_bits_h[i]);
		if (i != ((Qm * 12 * M_pusch_sc * N_l) - 1))
			printf(",");
	}
	printf(" ];\n");



	//RI Bits
	printf("\nRI_bits_cuda = [ ");
	for (int i = 0; i < (N_ri * Qm * N_l); i++)
	{
		printf("%d", ri_h[i]);
		if (i != ((N_ri * Qm * N_l) - 1))
			printf(",");
	}
	printf(" ];\n");

	//printf("N_id_cell = 2;N_sc_rb   = 12;M_pusch_rb = 100;M_pusch_sc = M_pusch_rb*N_sc_rb;Nc = 1600;n_s = 0;n_RNTI = 10;M_bits = 86400;N_l = 2;\nN_ri_bits = 0;N_ack_bits =0;Q_m = 6;\nmodulated_subframe = subframe_input_CUDA;\ndemodulated_subframe = sc_fdma_demodulator(modulated_subframe, M_pusch_rb);\ndemodulated_subframe_vect =[demodulated_subframe(0+1,:), demodulated_subframe(1+1,:), demodulated_subframe(2+1,:), demodulated_subframe(4+1,:), demodulated_subframe(5+1,:), demodulated_subframe(6+1,:), demodulated_subframe(7+1,:), demodulated_subframe(8+1,:), demodulated_subframe(9+1,:), demodulated_subframe(11+1,:), demodulated_subframe(12+1,:), demodulated_subframe(13+1,:)];\ndmrs = generate_dmrs_pusch(n_s, N_id_cell, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 0);\ndmrs_1 = dmrs(1:M_pusch_sc);\ndmrs_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);\ndmrs_1_rx = demodulated_subframe(1+3,:);\ndmrs_2_rx = demodulated_subframe(1+10,:);\npredecoded_data = transform_predecoder(demodulated_subframe_vect, M_pusch_rb);\n demapped_data = demapper_hard(predecoded_data, '64qam');\n c_init = n_RNTI * 2 ^ 14 + floor(n_s / 2) * 2 ^ 9 + N_id_cell;\n c = generate_psuedo_random_seq(c_init, M_bits);\n descrambled_bits = descramble(demapped_data, c);\n [data_bits, ri_bits, ack_bits] = channel_deinterleaver(descrambled_bits, N_ri_bits, N_ack_bits, N_l, Q_m); \nisequal(data_bits, Receved_bits_cuda)\nisequal(ri_bits, RI_bits_cuda)\n");
	printf("N_id_cell = 2;N_sc_rb   = 12;M_pusch_rb = 100;M_pusch_sc = M_pusch_rb*N_sc_rb;Nc = 1600;n_s = 0;n_RNTI = 10;M_bits = 86400;N_l = 2;\nN_ri_bits = 0;N_ack_bits =0;Q_m = 6;\nmodulated_subframe = subframe_input_CUDA;\ndemodulated_subframe = sc_fdma_demodulator_MIMO(modulated_subframe, M_pusch_rb, N_l);\nDecoded_streams = decompose_subframe_mimo(demodulated_subframe, N_l);\ntransform_predecoded_symbols = transform_predecoder_mimo(Decoded_streams.', M_pusch_rb, N_l);\n [layer1, layer2] = layer_demapping(transform_predecoded_symbols, N_l);\ndemapped_bits1 = transpose(demapper_hard(layer1, '64qam'));\ndemapped_bits2 = transpose(demapper_hard(layer2, '64qam'));\n c_init = n_RNTI * 2 ^ 14 + floor(n_s / 2) * 2 ^ 9 + N_id_cell;\n c = generate_psuedo_random_seq(c_init, M_bits);\n descrambled_bits = descrambler_MIMO([demapped_bits1 demapped_bits2].', [c; c], N_l);\n [data_bits, ri_bits, ack_bits] = channel_deinterleaver_MIMO(descrambled_bits.', 0, 0, N_l, Q_m); \nisequal(data_bits, Receved_bits_cuda)\nisequal(ri_bits, RI_bits_cuda)\n");
	//printf("sum(round(complex_data_h,6)-round(Decoded_streams(1,:),6))\n");
	//printf("sum(round(complex_data_h2,6)-round(Decoded_streams(2,:),6))\n");
	/*printf("isequal(descrambled_bits_cuda, descrambled_bits(1,:))\n");
	printf("isequal(descrambled_bits2_cuda, descrambled_bits(2,:))");*/

//	printf("sum(round(Receved_bits_cuda,6)-round(,6))\n");

	fclose(results1);
	return 0;
}