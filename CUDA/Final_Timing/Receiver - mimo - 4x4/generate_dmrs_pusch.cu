/*
% Function:    generate_dmrs_pusch
% Description: Generates LTE demodulation reference signal for PUSCH
% Inputs:      N_subfr                  - Subframe number within a radio frame
%              N_id_cell                - Physical layer cell identity
%              delta_ss                 - Configurable portion of the sequence-shift pattern for PUSCH (sib2 groupAssignmentPUSCH)
%              group_hopping_enabled    - Boolean value determining if group hopping is enabled (sib2 groupHoppingEnabled)
%              sequence_hopping_enabled - Boolean value determining if sequence hopping is enabled (sib2 sequenceHoppingEnabled)
%              cyclic_shift             - Broadcast cyclic shift to apply to base reference signal (sib2 cyclicShift)
%              cyclic_shift_dci         - Scheduled cyclic shift to apply to base reference signal
%              w_config                 - fixed or table
%              N_prbs                   - Number of PRBs used for the uplink grant
%              layer                    - Which diversity layer to generate reference signals for
% Outputs:     *dmrs1_h					- Demodulation reference signal for PUSCH
			   *dmrs2_h					- Demodulation reference signal for PUSCH
By: Mohammed Mostafa   
*/

#include "generate_dmrs_pusch.cuh"
#include "generate_ul_rs.cuh"
#include "generate_psuedo_random_seq.cuh"

__global__ void generate_reference_signal(cufftComplex* dmrs2_d, int w_vector, int M_sc_rb) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (x_idx >= M_sc_rb)
		return;

	dmrs2_d[x_idx] = w_vector * dmrs2_d[x_idx];
}


void generate_dmrs_pusch(int N_subfr, int N_id_cell, int delta_ss, bool group_hopping_enabled, bool sequence_hopping_enabled, int cyclic_shift, int cyclic_shift_dci, char* w_config, int N_prbs, int layer, cufftComplex** dmrs1_d, cufftComplex** dmrs2_d, cufftComplex* x_q_d)
{

	//Calculate M_sc_rb   (called in generate_ul_rs M_sc_rs)
	int M_sc_rb = N_prbs*N_sc_rb;

	//Calculate N_s
	int N_s = N_subfr * 2;

	//Set lambda
	int lambda = layer;

	//Calculate f_ss_pusch
	int f_ss_pusch = ((N_id_cell % 30) + delta_ss) % 30;

	//Generate c
	Byte* c = (Byte*)malloc(sizeof(Byte)* 8 * N_ul_symb * 20);
	int c_init = floor(N_id_cell / 30) * 32 + f_ss_pusch;
	generate_psuedo_random_seq(&c, 8 * N_ul_symb * 20, 0, 0, c_init);      //added c_init in N_id_cell according to ahmed nour
	
	//Calculate n_pn_ns
	int n_pn_ns_1 = c[8 * N_ul_symb*N_s + 0] + c[8 * N_ul_symb*N_s + 1] * 2 + c[8 * N_ul_symb*N_s + 2] * 4 + c[8 * N_ul_symb*N_s + 3] * 8 + c[8 * N_ul_symb*N_s + 4] * 16 + c[8 * N_ul_symb*N_s + 5] * 32 + c[8 * N_ul_symb*N_s + 6] * 64 + c[8 * N_ul_symb*N_s + 7] * 128;
	int n_pn_ns_2 =  c[8 * N_ul_symb*(N_s + 1) + 0] + c[8 * N_ul_symb*(N_s + 1) + 1]*2 + c[8 * N_ul_symb*(N_s + 1) + 2]*4 + c[8 * N_ul_symb*(N_s + 1) + 3]*8 + c[8 * N_ul_symb*(N_s + 1) + 4]*16 + c[8 * N_ul_symb*(N_s + 1) + 5]*32 + c[8 * N_ul_symb*(N_s + 1) + 6]*64 + c[8 * N_ul_symb*(N_s + 1) + 7]*128;
	
	//Determine n_1_dmrs
	int n_1_dmrs = N_1_DMRS[cyclic_shift];

	//Determine n_2_dmrs_lambda
	int n_2_dmrs_lambda = N_2_DMRS_LAMBDA[cyclic_shift_dci][lambda];

	//Calculate n_cs_lambda
	int n_cs_lambda_1 = (n_1_dmrs + n_2_dmrs_lambda + n_pn_ns_1) % 12;
	int n_cs_lambda_2 = (n_1_dmrs + n_2_dmrs_lambda + n_pn_ns_2) % 12;

	//Calculate alpha_lambda
	float alpha_lambda_1 = 2 * PI *n_cs_lambda_1 / (float)12;
	float alpha_lambda_2 = 2 * PI *n_cs_lambda_2 / (float)12;

	//Generate the base reference signal
	generate_ul_rs(N_s, N_id_cell, "pusch", delta_ss, group_hopping_enabled, sequence_hopping_enabled, alpha_lambda_1, N_prbs, &*dmrs1_d, x_q_d);
	generate_ul_rs(N_s+1, N_id_cell, "pusch", delta_ss, group_hopping_enabled, sequence_hopping_enabled, alpha_lambda_2, N_prbs, &*dmrs2_d, x_q_d);
	
	//Determine w vector
	int w_vector;
	if (strcmp(w_config, "fixed"))
	{
		w_vector = 1;
	}
	else
	{
		w_vector = W_VECTOR[cyclic_shift_dci*4 + lambda];
	}

	//Generate the PUSCH demodulation reference signal sequence
	generate_reference_signal << < 2, 1024 >> >(*dmrs2_d, w_vector, M_sc_rb);

}