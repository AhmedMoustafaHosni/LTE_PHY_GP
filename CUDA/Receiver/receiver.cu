/*
% Function:    receiver
% Inputs:      modulated_subframe       - modulated subframe (modulated symbols)
%              M_pusch_rb               - number of resource blocks assigned to the UE
% Outputs:     received_bits			- received bits
By: Mohammed Mostafa
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

int main(int argc, char **argv) {

	const int Qm = 2;				// Modulation Order(2 = QPSK, 4 = 16QAM, 6 = 64QAM)
	const int M_pusch_rb = 100;		//number of resource blocks assigned to the UE
	const int N_l = 1;				// Number of Layers
	const int N_ri = 12;			//length of ri symbols
	const int n_s = 0;				//assume UE send on subframe 0
	const int N_id_cell = 2;		//assume enodeB scheduled cell 2 for the UE
	const int M_pusch_sc = N_sc_rb * M_pusch_rb; //total number of subcarriers
	const int n_RNTI = 10;			//radio network temporary identifier given to the UE by enodeB (assume 10)
	const int N_bits = Qm * 12 * M_pusch_sc;   //Qm * 12 * M_pusch_sc = 2*12*1200

	//input
	cufftComplex* subframe_h = (cufftComplex *)malloc(sizeof(cufftComplex)*modulated_subframe_length);

	for (int i = 0; i < modulated_subframe_length; i++)
	{
		subframe_h[i].x = rand() / (float)RAND_MAX * 100;
		subframe_h[i].y = rand() / (float)RAND_MAX * 100;
	}


	cufftComplex* subframe_d;
	cudaMalloc((void **)&subframe_d, sizeof(cufftComplex)*modulated_subframe_length);
	cudaMemcpy(subframe_d, subframe_h, sizeof(cufftComplex)*modulated_subframe_length, cudaMemcpyHostToDevice);

	//Device data allocation
	cufftComplex* fft_vec_d;
	cufftComplex* demod_subframe_d;
	cudaMalloc((void **)&fft_vec_d, sizeof(cufftComplex)*N_symbs_per_subframe*FFT_size);
	cudaMalloc((void **)&demod_subframe_d, sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc);

	cufftComplex* x_q_d;
	cufftComplex* dmrs1_generated_d = 0, *dmrs2_generated_d = 0;
	cudaMalloc((void **)&dmrs1_generated_d, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&dmrs2_generated_d, sizeof(cufftComplex)*N_sc_rb*M_pusch_rb);
	cudaMalloc((void **)&x_q_d, sizeof(cufftComplex)*prime_nums[M_pusch_rb - 1]);

	cufftComplex* dmrs1_decomposed_d;
	cufftComplex* dmrs2_decomposed_d;
	cufftComplex* complex_data_d;
	cudaMalloc((void **)&complex_data_d, sizeof(cufftComplex) * 12 * M_pusch_sc);
	cudaMalloc((void **)&dmrs1_decomposed_d, sizeof(cufftComplex)*M_pusch_sc);
	cudaMalloc((void **)&dmrs2_decomposed_d, sizeof(cufftComplex)*M_pusch_sc);

	cufftComplex* predecoded_data_d;
	cudaMalloc((void **)&predecoded_data_d, sizeof(cufftComplex)* 12 * M_pusch_sc);

	Byte *bits_d;
	cudaMalloc((void **)&bits_d, sizeof(Byte)* Qm * 12 * M_pusch_sc);    //FIX Number_demaped_bits

	Byte* c_d = 0;
	cudaMalloc((void **)&c_d, sizeof(Byte)*Qm * 12 * M_pusch_sc);
	Byte *descrambled_bits_d;
	cudaMalloc((void **)&descrambled_bits_d, sizeof(Byte)* Qm * 12 * M_pusch_sc);


	// Step 1: Define C_mux
	int C_mux = N_pusch_symbs;
	// Step 2: Define R_mux and R_prime_mux
	int H_prime_total = N_bits / (Qm*N_l);
	int H_prime = H_prime_total - N_ri;
	int R_mux = (H_prime_total*Qm*N_l) / C_mux;
	int R_prime_mux = R_mux / (Qm*N_l);

	Byte *ri_d, *y_idx_d, *y_mat_d;
	Byte *received_bits_d;
	cudaMalloc((void **)&ri_d, sizeof(Byte)*(N_ri * Qm * N_l));
	cudaMalloc((void **)&y_idx_d, sizeof(Byte)*(C_mux*R_prime_mux));
	cudaMalloc((void **)&y_mat_d, sizeof(Byte)*(C_mux*R_mux));
	cudaMalloc((void **)&received_bits_d, sizeof(Byte)* H_prime * Qm * N_l);

	//create plans
	int n[1] = { FFT_size };
	cufftHandle plan_sc_fdma;
	cufftPlanMany(&plan_sc_fdma, 1, n, NULL, 1, FFT_size, NULL, 1, FFT_size, CUFFT_C2C, N_symbs_per_subframe);

	int N_SIGS = 12;   //signal_size/M_pusch_sc = 12 * M_pusch_sc / M_pusch_sc = 12
	n[0] = { M_pusch_sc };
	cufftHandle plan_transform_predecoder;
	cufftPlanMany(&plan_transform_predecoder, 1, n, NULL, 1, M_pusch_sc, NULL, 1, M_pusch_sc, CUFFT_C2C, N_SIGS);
	
	//sc-fdma demodulation
	sc_fdma_demodulator(subframe_d, M_pusch_rb, &demod_subframe_d, plan_sc_fdma, fft_vec_d);

	//generate dmrs   
	generate_dmrs_pusch(n_s, N_id_cell, 0, 0, 0, 0, 0, "fixed", M_pusch_rb, 0, &dmrs1_generated_d, &dmrs2_generated_d, x_q_d);

	//Decompose subframe
	decompose_subframe(demod_subframe_d, M_pusch_rb, &complex_data_d, &dmrs1_decomposed_d, &dmrs2_decomposed_d);

	//FIX  "CHANNEL ESTIMATION" SECTIONS
	

	//predecoding   
	transform_predecoder(complex_data_d, M_pusch_rb, 12 * M_pusch_sc , &predecoded_data_d, plan_transform_predecoder); //signal_size = 12 * M_pusch_sc

	//demapping
	demapper(predecoded_data_d, &bits_d, Qm * 12 * M_pusch_sc , Qm);  //Number_demaped_bits = Qm * 12 * M_pusch_sc

	
	//Generate Pseudo Random Seq.
	Byte *c_h = 0;
	generate_psuedo_random_seq(&c_h, N_bits, n_RNTI, n_s, N_id_cell);
	//Copy (c) to Device
	cudaMemcpy(c_d, c_h, sizeof(Byte)*N_bits, cudaMemcpyHostToDevice);
	//Descrammpling
	descrambler(bits_d, &descrambled_bits_d, c_d, N_bits);

	//deinterleaver
	deinterleaver(descrambled_bits_d, &ri_d, &received_bits_d, N_bits, N_ri, Qm, N_l, y_idx_d, y_mat_d);

	//Retrieve data from device
	Byte* received_bits_h = (Byte*)malloc(sizeof(Byte)*H_prime * Qm * N_l);
	cudaMemcpy(received_bits_h, received_bits_d, sizeof(Byte)*H_prime * Qm * N_l, cudaMemcpyDeviceToHost);
	Byte* ri_h = (Byte*)malloc(sizeof(Byte)*N_ri * Qm * N_l);
	cudaMemcpy(ri_h, ri_d, sizeof(Byte)*N_ri * Qm * N_l, cudaMemcpyDeviceToHost);

	//Print results
	for (int i = 0; i < H_prime * Qm * N_l; i++)
	{
			printf("idx = %d \t %d  \n", i + 1, received_bits_h[i]);
	}
	

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
	printf("subframe_input_CUDA = symbols_in_real + 1i * symbols_in_imag;\n");

	//Received Bits
	printf("\nReceved_bits_cuda = [ ");
	for (int i = 0; i < (H_prime * Qm * N_l); i++)
	{
		printf("%d", received_bits_h[i]);
		if (i != ((Qm * 12 * M_pusch_sc)-1))
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

	printf("N_id_cell = 2;N_sc_rb   = 12;M_pusch_rb = 100;M_pusch_sc = M_pusch_rb*N_sc_rb;Nc = 1600;n_s = 0;n_RNTI = 10;M_bits = 2*1200*12;N_l = 1;N_ri_bits = 12;N_ack_bits =0;Q_m = 2;modulated_subframe = subframe_input_CUDA;demodulated_subframe = sc_fdma_demodulator(modulated_subframe, M_pusch_rb);demodulated_subframe_vect =[demodulated_subframe(0+1,:), demodulated_subframe(1+1,:), demodulated_subframe(2+1,:), demodulated_subframe(4+1,:), demodulated_subframe(5+1,:), demodulated_subframe(6+1,:), demodulated_subframe(7+1,:), demodulated_subframe(8+1,:), demodulated_subframe(9+1,:), demodulated_subframe(11+1,:), demodulated_subframe(12+1,:), demodulated_subframe(13+1,:)];dmrs = generate_dmrs_pusch(n_s, N_id_cell, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 0);dmrs_1 = dmrs(1:M_pusch_sc);dmrs_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);dmrs_1_rx = demodulated_subframe(1+3,:);dmrs_2_rx = demodulated_subframe(1+10,:);predecoded_data = transform_predecoder(demodulated_subframe_vect, M_pusch_rb);demapped_data = demapper_hard(predecoded_data, 'qpsk' );c_init = n_RNTI * 2^14 + floor(n_s/2) * 2^9 + N_id_cell;c = generate_psuedo_random_seq(c_init, M_bits);descrambled_bits = descramble(demapped_data, c);[data_bits, ri_bits, ack_bits] = channel_deinterleaver(descrambled_bits, N_ri_bits, N_ack_bits, N_l, Q_m);isequal(data_bits,Receved_bits_cuda)\nisequal(ri_bits, RI_bits_cuda)");

	fclose(results1);
	return 0;
}